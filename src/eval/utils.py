from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
from collections import namedtuple
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Array4D, Dataset, Features
from huggingface_hub import hf_hub_download
from omegaconf import DictConfig, OmegaConf
from sklearn import metrics as sk_metrics
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def classification_metrics(targets, predictions, probabilities) -> dict[str, float]:
    return {
        "accuracy": float(sk_metrics.accuracy_score(targets, predictions)),
        "f1_macro": float(sk_metrics.f1_score(targets, predictions, average="macro")),
        "balanced_accuracy": float(sk_metrics.balanced_accuracy_score(targets, predictions)),
        "auroc": float(sk_metrics.roc_auc_score(targets, probabilities)),
    }


def regression_metrics(targets, predictions) -> dict[str, float]:
    targets = np.asarray(targets); predictions = np.asarray(predictions)
    multi = targets.ndim == 2 and targets.shape[1] > 1
    result = {
        "mae": float(sk_metrics.mean_absolute_error(targets, predictions)),
        "rmse": float(np.sqrt(sk_metrics.mean_squared_error(targets, predictions))),
        "r2": float(sk_metrics.r2_score(targets, predictions, multioutput="uniform_average")),
    }
    if multi:
        correlations = [_pearson(targets[:, i], predictions[:, i]) for i in range(targets.shape[1])]
        result["pearson_r"] = float(np.nanmean(correlations))
    else:
        result["pearson_r"] = _pearson(targets.reshape(-1), predictions.reshape(-1))
        result["bias"] = float(np.mean(predictions - targets))
    return result


def per_output_regression_metrics(targets, predictions, names) -> list[dict]:
    targets = np.asarray(targets); predictions = np.asarray(predictions)
    return [{"target": name, **regression_metrics(targets[:, i], predictions[:, i])}
            for i, name in enumerate(names)]


def _pearson(x, y) -> float:
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def participant_bootstrap(metric_fn, targets, predictions, participant_ids, *, seed=4466,
                          repetitions=500) -> dict[str, float]:
    participant_ids = np.asarray(participant_ids)
    unique = np.unique(participant_ids)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repetitions):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(participant_ids == p) for p in sampled])
        try:
            values.append(float(metric_fn(np.asarray(targets)[indices], np.asarray(predictions)[indices])))
        except ValueError:
            continue
    return {"bootstrap_mean": float(np.mean(values)), "bootstrap_std": float(np.std(values))}


# ---------------------------------------------------------------------------
# Data preprocessing (dataset-agnostic)
# ---------------------------------------------------------------------------

# Returned by every dataset builder alongside its preprocessed splits, so the eval
# mains know how to build the head/metrics without knowing the dataset.
Task = namedtuple("Task", ["type", "output_dim", "target"])

# Raw columns every dataset provides and that _normalize consumes.
RAW_COLUMNS = ("nifti", "mask")


def pad_to_shape(x: torch.Tensor, target_shape: tuple[int, ...]) -> torch.Tensor:
    # nb this also crops
    padding = []
    for s, s_ in reversed(list(zip(x.shape, target_shape))):
        pad = s_ - s
        padding.extend([pad // 2, pad - pad // 2])
    return F.pad(x, padding)


def _normalize(example, img_size):
    """z-score over the brain mask + pad/crop to img_size; emit fp16 image + mask.

    Dataset-agnostic: depends only on the raw `nifti` scan and `mask`. Scans are
    already RAS-oriented and 1mm isotropic, so no reorient/resample is needed.
    """
    data = torch.from_numpy(example["nifti"].get_fdata(dtype=np.float32))
    mask = torch.from_numpy(example["mask"].get_fdata(dtype=np.float32)) > 0.5

    # z-score over brain-mask voxels (matches pretraining); background -> 0.
    # Raw intensities reach ~1e6, so this must happen before the fp16 cast.
    brain = data[mask]
    # population std (÷N, correction=0) to match the pretraining normalization.
    mean, std = brain.mean(), brain.std(correction=0).clamp_min(1e-6)
    data = torch.where(mask, (data - mean) / std, torch.zeros_like(data))

    # (X, Y, Z) F-order -> (Z, Y, X) C-order, pad/crop to img_size, add channel dim.
    image = pad_to_shape(data.permute(2, 1, 0).contiguous(), img_size).half().unsqueeze(0)
    mask = pad_to_shape(mask.permute(2, 1, 0).contiguous().float(), img_size).unsqueeze(0) > 0.5
    return {"image": image.numpy(), "mask": mask.numpy()}


def map_normalize(dataset: Dataset, img_size: tuple[int, int, int], num_proc: int,
                  *, drop: tuple[str, ...] = ()) -> Dataset:
    """Apply _normalize once and persist it in the HF Arrow cache.

    Dataset-agnostic: the only universal outputs are fp16 `image`/`mask`; every other
    column (metadata + label) is carried through with its original type, so the schema
    isn't hardcoded. Target-agnostic too, so the cache is keyed only on img_size and
    shared across every prediction target of a dataset. Columns in `drop` are removed.
    """
    missing = set(RAW_COLUMNS) - set(dataset.column_names)
    if missing:
        raise ValueError(f"dataset is missing raw columns: {sorted(missing)}")

    carry = [c for c in dataset.column_names if c not in RAW_COLUMNS and c not in drop]
    features = Features({
        **{c: dataset.features[c] for c in carry},
        "image": Array4D(shape=(1, *img_size), dtype="float16"),
        "mask": Array4D(shape=(1, *img_size), dtype="uint8"),
    })
    remove_columns = [c for c in dataset.column_names if c not in carry]

    # Array4D image cells are large (~25MB at 208x240x208 fp16). Arrow lists use
    # 32-bit offsets, so a write batch whose image column exceeds 2GB overflows
    # during .map concatenation. Cap the batch so the image column stays ~1GB.
    bytes_per_image = int(np.prod(img_size)) * 2
    writer_batch_size = max(1, min(1000, int(1e9 // bytes_per_image)))

    return dataset.map(
        _normalize, fn_kwargs={"img_size": img_size},
        remove_columns=remove_columns, features=features,
        num_proc=num_proc if num_proc > 1 else None,
        writer_batch_size=writer_batch_size,
        new_fingerprint=f"{dataset._fingerprint}-normalize-{'x'.join(map(str, img_size))}",
        desc="Preprocessing",
    )


# ---------------------------------------------------------------------------
# Shared loader
# ---------------------------------------------------------------------------

def make_loader(dataset, cfg, *, shuffle=False):
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        prefetch_factor=cfg.get("prefetch_factor") if cfg.num_workers else None,
        drop_last=shuffle,
    )


# ---------------------------------------------------------------------------
# Config and shared helpers
# ---------------------------------------------------------------------------

def load_config(default_path: Path, args: argparse.Namespace, positional: dict[str, str]) -> DictConfig:
    cfg = OmegaConf.load(default_path)
    if args.config:
        cfg = OmegaConf.unsafe_merge(cfg, OmegaConf.load(args.config))
    if args.overrides:
        cfg = OmegaConf.unsafe_merge(cfg, OmegaConf.from_dotlist(args.overrides))
    for key, value in positional.items():
        cfg[key] = value
    return cfg


def select_representation(embeddings, representation: str):
    value = getattr(embeddings, representation)
    if value is None:
        raise ValueError(f"representation {representation!r} is unavailable")
    return value.mean(dim=1) if value.ndim == 3 else value


def send_batch(batch, device: torch.device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def token_from_config(cfg: DictConfig):
    return cfg.dataset_kwargs.get("token") or os.getenv("HF_TOKEN")


# ---------------------------------------------------------------------------
# Training helpers (grid probe)
# ---------------------------------------------------------------------------

def random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def update_lr(param_groups, lr: float) -> None:
    for group in param_groups:
        group["lr"] = lr * group.get("lr_multiplier", 1.0)


def update_wd(param_groups, weight_decay: float) -> None:
    for group in param_groups:
        group["weight_decay"] = weight_decay * group.get("wd_multiplier", 1.0)


def make_lr_schedule(base_lr: float, total_steps: int, warmup_steps: int,
                     no_decay: bool = False) -> np.ndarray:
    warmup = np.linspace(0.0, 1.0, warmup_steps) if warmup_steps > 0 else np.empty(0)
    decay_steps = max(total_steps - warmup_steps, 0)
    if no_decay:
        decay = np.ones(decay_steps)
    else:
        decay = (np.cos(np.linspace(0, np.pi, decay_steps)) + 1) / 2
    schedule = base_lr * np.concatenate([warmup, decay])
    return schedule[:total_steps]


def infinite_loader(loader):
    while True:
        yield from loader


def load_volume_names(
    repo_id: str,
    *,
    cache_dir: str | Path | None = None,
) -> list[str]:
    path = hf_hub_download(
        repo_id, "synthseg_volume_names.json", repo_type="dataset",
        token=os.getenv("HF_TOKEN"), cache_dir=str(cache_dir) if cache_dir else None,
    )
    names = json.loads(Path(path).read_text())
    if len(names) != 101:
        raise ValueError(f"expected 101 SynthSeg volume names, got {len(names)}")
    return names
