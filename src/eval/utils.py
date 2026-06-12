from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
from datasets import Array4D, DatasetDict, Features, Value
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
# Data preprocessing
# ---------------------------------------------------------------------------

KEEP_COLUMNS = [
    "sample_id", "participant_id", "age", "sex", "diagnosis", "synthseg_volumes",
    "train_rank",
]


def map_model_transform(dataset: DatasetDict, transform, *, checkpoint_path: str,
                        num_proc: int = 4) -> DatasetDict:
    """Apply model preprocessing once and persist it in the HF Arrow cache."""
    signature = _transform_signature(transform, checkpoint_path)
    mapped = {}
    for split, split_dataset in dataset.items():
        missing = set(KEEP_COLUMNS + ["nifti", "mask"]) - set(split_dataset.column_names)
        if missing:
            raise ValueError(f"dataset is missing preprocessing columns: {sorted(missing)}")
        features = Features({
            "sample_id": Value("string"), "participant_id": Value("string"),
            "age": Value("float32"), "sex": Value("int64"),
            "diagnosis": Value("int64"),
            "synthseg_volumes": split_dataset.features["synthseg_volumes"],
            "train_rank": Value("int32"),
            "image": Array4D(shape=(1, *transform.img_size), dtype="float32"),
            "mask": Array4D(shape=(1, *transform.img_size), dtype="uint8"),
        })
        remove_columns = [c for c in split_dataset.column_names if c not in KEEP_COLUMNS]
        mapped[split] = split_dataset.map(
            _apply_transform, fn_kwargs={"transform": transform},
            remove_columns=remove_columns, features=features,
            num_proc=num_proc if num_proc > 1 else None,
            new_fingerprint=f"{split_dataset._fingerprint}-transform-{signature}",
            desc=f"Preprocessing {split} for model",
        )
    return DatasetDict(mapped)


def _transform_signature(transform, checkpoint_path: str) -> str:
    checkpoint = Path(checkpoint_path)
    stat = checkpoint.stat()
    payload = (
        type(transform).__module__, type(transform).__qualname__, repr(vars(transform)),
        str(checkpoint.resolve()), stat.st_size, stat.st_mtime_ns,
    )
    return hashlib.sha256(repr(payload).encode()).hexdigest()[:16]


def _apply_transform(example, transform):
    sample = transform(example["nifti"], example["mask"])
    image = sample["image"]
    mask = sample["mask"]
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if mask.ndim == 3:
        mask = mask.unsqueeze(0)
    return {"image": image.numpy(), "mask": mask.to("cpu").numpy()}


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


def prepare_datasets(cfg: DictConfig, datasets, transform, checkpoint_path: str):
    """Apply the model transform to wrapped task datasets in place.

    `datasets` maps split -> task dataset (e.g. ADNIDataset), each holding a raw
    HF dataset on `.dataset`. The model transform is precomputed and cached, then
    swapped back into each wrapped dataset. Returns (datasets, train_dataset);
    callers read `.kind` / `.output_dim` off the train dataset.
    """
    raw = DatasetDict({split: ds.dataset for split, ds in datasets.items()})
    train_size = cfg.get("train_size")
    if train_size:
        raw["train"] = raw["train"].filter(
            lambda rank: rank < train_size, input_columns="train_rank"
        )
    mapped = map_model_transform(
        raw,
        transform,
        checkpoint_path=checkpoint_path,
        num_proc=cfg.map_workers,
    )
    for split, ds in datasets.items():
        ds.with_images(mapped[split])
    return datasets, datasets["train"]


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
