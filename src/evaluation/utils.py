from functools import lru_cache
import logging
import random
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from datasets import Dataset as HFDataset
from nibabel.processing import resample_from_to
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@lru_cache(maxsize=1)
def load_mni_brain_mask() -> nib.Nifti1Image:
    """The 1mm MNI152NLin2009cAsym brain mask (reoriented to RAS), loaded once."""
    import templateflow.api as tflow

    path = tflow.get(
        "MNI152NLin2009cAsym",
        resolution=1,
        desc="brain",
        suffix="mask",
        extension=".nii.gz",
    )
    return nib.as_closest_canonical(nib.load(str(path)))


def resample_binary_mask(
    mask_image: nib.Nifti1Image,
    target_image: nib.Nifti1Image,
) -> np.ndarray:
    """Resample a binary mask to an image grid with nearest-neighbor interpolation."""
    if (
        mask_image.shape == target_image.shape
        and np.allclose(mask_image.affine, target_image.affine)
    ):
        return np.asanyarray(mask_image.dataobj) > 0
    resampled = resample_from_to(mask_image, target_image, order=0)
    return np.asanyarray(resampled.dataobj) > 0


def build_covariates(data: HFDataset, columns: Sequence[str]) -> np.ndarray:
    """Design matrix of nuisance covariates, aligned to ``data`` row order.

    A quadratic term is added for ``age`` so a covariate-only floor can model the
    curved age-vs-biomarker relationship (a linear floor would be handicapped and
    the image would be wrongly credited for the curvature).
    """
    feats = []
    for name in columns:
        col = np.asarray(data[name], dtype=np.float64).reshape(len(data), -1)
        feats.append(col)
        if name == "age":
            feats.append(col**2)
    return np.hstack(feats)


# Estimators, keyed by task kind. Hyperparameters are selected by inner CV.
def fit_ridge(X: np.ndarray, y: np.ndarray, seed: int) -> Pipeline:
    ridge = RidgeCV(alphas=np.logspace(-3, 3, 13))
    model = Pipeline([("scaler", StandardScaler()), ("ridge", ridge)])
    return model.fit(X, y)


def fit_logistic(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    selection_metric: str = "balanced_accuracy",
    Cs: int | list[float] = 10,
    max_iter: int = 1000,
) -> Pipeline:
    clf = LogisticRegressionCV(
        Cs=Cs,
        scoring=selection_metric,
        max_iter=max_iter,
        random_state=seed,
    )
    model = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    return model.fit(X, y)


ESTIMATORS = {"regression": fit_ridge, "classification": fit_logistic}


def fit_score(
    task: Any,
    features: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
    estimator_kwargs: Mapping[str, Any] | None = None,
):
    """Fit a task estimator on ``features``; return (estimator, pred, y_score)."""
    fit = ESTIMATORS[task.kind]
    estimator_kwargs = dict(estimator_kwargs or {})
    if task.kind == "classification":
        estimator = fit(
            features[train_idx],
            y[train_idx],
            seed,
            getattr(task, "selection_metric", "balanced_accuracy"),
            **estimator_kwargs,
        )
    else:
        estimator = fit(features[train_idx], y[train_idx], seed, **estimator_kwargs)

    pred = estimator.predict(features[test_idx])
    y_score = None
    if task.kind == "classification":
        y_score = classification_score(
            estimator, features[test_idx], getattr(task, "positive_label", None)
        )
    return estimator, pred, y_score


def _json_key(value: Any) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    return str(value)


def _value_counts(values: np.ndarray) -> dict[str, int]:
    return {_json_key(key): int(value) for key, value in Counter(values).items()}


def task_group_values(task: Any) -> np.ndarray | None:
    group_column = getattr(task, "group_column", None)
    data = getattr(task, "data", None)
    if not group_column or data is None:
        return None
    return np.asarray(data[group_column])


def task_data_summary(task: Any, y: np.ndarray, groups: np.ndarray | None) -> dict:
    participant_level = bool(getattr(task, "participant_level", False))
    summary = {
        "participant_level": participant_level,
        "metric_level": "participant" if participant_level else "session",
        "n_scans": int(len(y)),
    }
    if groups is not None:
        summary["n_participants"] = int(len(set(groups)))
    if task.kind == "classification":
        counts = _value_counts(y)
        total = sum(counts.values())
        summary["class_counts"] = counts
        summary["class_prevalence"] = {key: value / total for key, value in counts.items()}
    return summary


def fold_data_summary(
    task: Any,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    groups: np.ndarray | None,
) -> dict:
    summary = {
        "train_scans": int(len(train_idx)),
        "test_scans": int(len(test_idx)),
    }
    if groups is not None:
        summary["train_participants"] = int(len(set(groups[train_idx])))
        summary["test_participants"] = int(len(set(groups[test_idx])))
    if task.kind == "classification":
        summary["train_class_counts"] = _value_counts(y[train_idx])
        summary["test_class_counts"] = _value_counts(y[test_idx])
    return summary


def _mode(values: list[Any]) -> Any:
    return Counter(values).most_common(1)[0][0]


def _participant_aggregate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    y_score: np.ndarray | None,
    *,
    kind: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict[str, int]]:
    df = pd.DataFrame(
        {
            "group": groups,
            "y_true": list(y_true),
            "y_pred": list(y_pred),
        }
    )
    if y_score is not None:
        df["y_score"] = list(y_score)

    agg_true = []
    agg_pred = []
    agg_score = []
    dropped_conflicting = 0
    for _, participant in df.groupby("group", sort=False):
        true = participant["y_true"].tolist()
        pred = participant["y_pred"].tolist()
        if kind == "classification":
            if len(set(true)) != 1:
                dropped_conflicting += 1
                continue
            agg_true.append(true[0])
            agg_pred.append(_mode(pred))
        else:
            agg_true.append(np.mean(np.asarray(true, dtype=np.float64), axis=0))
            agg_pred.append(np.mean(np.asarray(pred, dtype=np.float64), axis=0))
        if y_score is not None:
            agg_score.append(np.mean(np.asarray(participant["y_score"].tolist()), axis=0))

    score_array = np.asarray(agg_score) if y_score is not None else None
    metadata = {
        "participant_metric_count": int(len(agg_true)),
        "participant_metric_dropped_conflicting_labels": int(dropped_conflicting),
    }
    return np.asarray(agg_true), np.asarray(agg_pred), score_array, metadata


def compute_task_scores(
    task: Any,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    test_idx: np.ndarray,
    y_score: np.ndarray | None,
    *,
    level: str,
    groups: np.ndarray | None,
) -> tuple[dict[str, float], dict[str, int]]:
    if level == "session":
        return task.compute_metrics(y_true, y_pred, test_idx, y_score=y_score), {}
    if level != "participant":
        raise ValueError(f"unsupported metric level {level!r}")
    if groups is None:
        raise ValueError(f"{task.name} requested participant metrics without group_column")
    agg_true, agg_pred, agg_score, metadata = _participant_aggregate(
        y_true,
        y_pred,
        groups[test_idx],
        y_score,
        kind=task.kind,
    )
    scores = task.compute_metrics(
        agg_true,
        agg_pred,
        np.arange(len(agg_true)),
        y_score=agg_score,
    )
    return scores, metadata


def classification_score(estimator: Pipeline, X: np.ndarray, positive_label) -> np.ndarray | None:
    if not hasattr(estimator, "predict_proba"):
        return None
    proba = estimator.predict_proba(X)
    if positive_label is None:
        return proba  # full matrix for multiclass metrics
    classes = np.asarray(estimator.classes_)
    matches = np.flatnonzero(classes == positive_label)
    if len(matches) != 1:
        raise ValueError(
            f"positive label {positive_label!r} not found in classes {classes.tolist()}"
        )
    return proba[:, matches[0]]


def to_device(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) for key, value in batch.items()}


def aggregate_folds(fold_metrics: list[dict[str, float]]) -> dict[str, float]:
    summary = {}
    for key in fold_metrics[0]:
        values = np.array([fold[key] for fold in fold_metrics])
        summary[key] = float(values.mean())
        summary[f"{key}_std"] = float(values.std())
    return summary


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
