import argparse
import json
import logging
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from evaluation.models.base import Model, Transform
from evaluation.models.registry import create_model, list_models
from evaluation.tasks.base import Task
from evaluation.tasks.registry import create_task, list_tasks
from evaluation.utils import aggregate_folds, fit_score, set_seed, to_device

DEFAULT_CONFIG = Path(__file__).parent / "config/default_linear.yaml"

logger = logging.getLogger(__name__)


class TransformDataset(Dataset):
    """Applies the model transform to each canonical sample for batched loading."""

    def __init__(self, dataset: Dataset, transform: Transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        sample = self.dataset[index]
        target = sample["target"]
        # Tensorize numeric targets so default collation stacks them into a (B, ...)
        # batch. Without this, a vector target (e.g. the 101 SynthSeg volumes) is a
        # Python list, which default collate transposes into per-element tensors and
        # scrambles targets relative to features. String class labels must remain
        # strings for sklearn classifiers.
        if not isinstance(target, str):
            target = torch.as_tensor(target)
        out = self.transform(sample["image"])
        out["target"] = target
        return out


@torch.inference_mode()
def extract_features(
    model: Model,
    dataset: Dataset,
    transform: Transform,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(
        TransformDataset(dataset, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    model.eval()

    features = []
    targets = []
    for batch in loader:
        targets.extend(batch.pop("target"))
        batch = to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            embeddings = model(batch)
        features.append(embeddings.cpu().float())

    X = torch.cat(features).numpy()
    y = np.asarray(targets)
    return X, y


def run_linear(
    task: Task,
    model: Model,
    transform: Transform,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    seed: int,
    estimator_kwargs: Mapping[str, Mapping[str, Any]] | None = None,
):
    logger.info("extracting features...")
    dataset = task.dataset()
    start = time.perf_counter()
    X, y = extract_features(model, dataset, transform, device, batch_size, num_workers)
    tput = len(y) / (time.perf_counter() - start)
    logger.info(f"features: {tuple(X.shape)} ({tput:.2f} samples/s)")

    fit_kwargs = dict((estimator_kwargs or {}).get(task.kind, {}))
    covariates = task.covariates() if hasattr(task, "covariates") else None
    combined_features = None if covariates is None else np.hstack([X, covariates])

    fold_metrics = []
    for fold, (train_idx, test_idx) in enumerate(task.split()):
        estimator, pred, y_score = fit_score(task, X, y, train_idx, test_idx, seed, fit_kwargs)
        postprocess = getattr(task, "postprocess_predictions", None)
        if postprocess is not None:
            pred = postprocess(
                y[train_idx],
                estimator.predict(X[train_idx]),
                y[test_idx],
                pred,
            )
        scores = task.compute_metrics(y[test_idx], pred, test_idx, y_score=y_score)

        # Age+sex floor: fit a covariate-only model (A) and a covariate+latent model
        # (B) on the same fold; report A, B, and the latent's contribution gap = B - A.
        if covariates is not None:
            _, floor_pred, floor_score = fit_score(
                task, covariates, y, train_idx, test_idx, seed, fit_kwargs
            )
            floor = task.compute_metrics(y[test_idx], floor_pred, test_idx, y_score=floor_score)
            _, comb_pred, comb_score = fit_score(
                task, combined_features, y, train_idx, test_idx, seed, fit_kwargs
            )
            combined = task.compute_metrics(y[test_idx], comb_pred, test_idx, y_score=comb_score)
            for key in floor:
                scores[f"floor_{key}"] = floor[key]
                scores[f"combined_{key}"] = combined[key]
                scores[f"gap_{key}"] = combined[key] - floor[key]

        fold_metrics.append(scores)
        logger.info(f"fold {fold}: " + " ".join(f"{k}={v:.4f}" for k, v in scores.items()))

    metrics = {
        "tput": tput,
        "model_selection_eligible": bool(getattr(task, "model_selection_eligible", True)),
        "summary": aggregate_folds(fold_metrics),
        "folds": fold_metrics,
    }
    return metrics


def setup_logging(run_dir: Path) -> None:
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(run_dir / "log.txt")]
    formatter = logging.Formatter("%(message)s")

    root = logging.getLogger("evaluation")
    root.setLevel(logging.INFO)
    root.handlers.clear()
    for handler in handlers:
        handler.setFormatter(formatter)
        root.addHandler(handler)
    root.propagate = False


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def main(
    model_name: str,
    task_name: str,
    config_path: str | Path | None = None,
    overrides: list[str] | None = None,
) -> dict:
    cfg = OmegaConf.load(DEFAULT_CONFIG)
    if config_path:
        cfg = OmegaConf.unsafe_merge(cfg, OmegaConf.load(config_path))
    if overrides:
        cfg = OmegaConf.unsafe_merge(cfg, OmegaConf.from_dotlist(overrides))

    set_seed(cfg.seed)

    cfg.model = model_name
    cfg.task = task_name
    cfg.name = cfg.name or f"{cfg.model}__{cfg.task}"

    run_dir = Path(cfg.output_root) / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir)

    logger.info(f"run: {cfg.name}")
    logger.info(f"git sha: {git_sha()}")
    logger.info(f"config:\n{OmegaConf.to_yaml(cfg).rstrip()}")
    OmegaConf.save(cfg, run_dir / "config.yaml")

    device = torch.device(cfg.device)

    task = create_task(cfg.task, **(cfg.task_kwargs or {}))
    model, transform = create_model(cfg.model, **(cfg.model_kwargs or {}))
    model.to(device)

    logger.info(f"task: {cfg.task} ({task.kind})")
    logger.info(f"model: {cfg.model}")
    logger.info(f"dataset: {len(task.dataset())} samples")

    metrics = run_linear(
        task,
        model,
        transform,
        device=device,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
        estimator_kwargs=OmegaConf.to_container(cfg.estimator_kwargs or {}, resolve=True),
    )

    metrics = {"model": cfg.model, "task": cfg.task, **metrics}
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    summary = pd.DataFrame(
        [{"model": cfg.model, "task": cfg.task, "tput": metrics["tput"], **metrics["summary"]}]
    )
    summary.to_csv(run_dir / "summary.csv", index=False)
    logger.info(f"summary:\n\n{summary.to_markdown(index=False, floatfmt='.4f')}")
    return metrics


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help=f"[{', '.join(list_models())}]")
    parser.add_argument("task", type=str, help=f"[{', '.join(list_tasks())}]")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--overrides", type=str, default=None, nargs="+")
    args = parser.parse_args()
    main(args.model, args.task, args.config, args.overrides)


if __name__ == "__main__":
    cli()
