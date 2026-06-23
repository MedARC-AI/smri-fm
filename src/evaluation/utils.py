from functools import lru_cache
import random
from collections.abc import Mapping, Sequence
from typing import Any

import nibabel as nib
import numpy as np
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
