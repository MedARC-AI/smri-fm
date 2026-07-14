import argparse
import json
import logging
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from evaluation.models.base import Model, Transform
from evaluation.models.registry import create_model, list_models
from evaluation.tasks.base import Task
from evaluation.tasks.metrics import classification_score
from evaluation.tasks.registry import create_task, list_tasks

DEFAULT_CONFIG = Path(__file__).parent / "config/default_linear.yaml"

logger = logging.getLogger(__name__)


# Estimators, keyed by task kind. Hyperparameters are selected by inner CV.
def fit_ridge(X: np.ndarray, y: np.ndarray, seed: int) -> Pipeline:
    ridge = RidgeCV(alphas=np.logspace(-3, 3, 13))
    model = Pipeline([("scaler", StandardScaler()), ("ridge", ridge)])
    return model.fit(X, y)


def fit_logistic(X: np.ndarray, y: np.ndarray, seed: int) -> Pipeline:
    clf = LogisticRegressionCV(Cs=10, scoring="balanced_accuracy", max_iter=1000, random_state=seed)
    model = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    return model.fit(X, y)


ESTIMATORS = {"regression": fit_ridge, "classification": fit_logistic}


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
        if not isinstance(target, str):
            target = torch.as_tensor(target)
        out = self.transform(sample["image"])
        out["target"] = target
        return out


def to_device(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) for key, value in batch.items()}


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
):
    logger.info("extracting features...")
    dataset = task.dataset()
    logger.info(f"dataset: {len(dataset)} samples")
    start = time.perf_counter()
    X, y = extract_features(model, dataset, transform, device, batch_size, num_workers)
    tput = len(y) / (time.perf_counter() - start)
    logger.info(f"features: {tuple(X.shape)} ({tput:.2f} samples/s)")

    fit = ESTIMATORS[task.kind]

    fold_metrics = []
    for fold, (train_idx, test_idx) in enumerate(task.split()):
        estimator = fit(X[train_idx], y[train_idx], seed)
        pred = estimator.predict(X[test_idx])
        y_score = None
        positive_label = getattr(task, "positive_label", None)
        if task.kind == "classification" and positive_label is not None:
            y_score = classification_score(
                estimator,
                X[test_idx],
                positive_label,
            )
        scores = task.metrics(y[test_idx], pred, test_idx, y_score=y_score)
        fold_metrics.append(scores)
        logger.info(f"fold {fold}: " + " ".join(f"{k}={v:.4f}" for k, v in scores.items()))

    metrics = {"tput": tput, "summary": aggregate_folds(fold_metrics), "folds": fold_metrics}
    return metrics


def aggregate_folds(fold_metrics: list[dict[str, float]]) -> dict[str, float]:
    summary = {}
    for key in fold_metrics[0]:
        values = np.array([fold[key] for fold in fold_metrics])
        summary[key] = float(values.mean())
        summary[f"{key}_std"] = float(values.std())
    return summary


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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

    task_kwargs = OmegaConf.to_container(cfg.task_kwargs, resolve=True)
    task = create_task(cfg.task, default_seed=cfg.seed, **task_kwargs)
    model, transform = create_model(cfg.model, **(cfg.model_kwargs or {}))
    model.to(device)

    logger.info(f"task: {cfg.task} ({task.kind})")
    logger.info(f"model: {cfg.model}")

    metrics = run_linear(
        task,
        model,
        transform,
        device=device,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
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
