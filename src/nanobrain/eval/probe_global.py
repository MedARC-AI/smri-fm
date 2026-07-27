"""Global-embedding probes: one pooled vector per volume -> sklearn head.

Feature extraction (GPU) is separated from scoring (sklearn/CPU). Regression uses RidgeCV,
classification LogisticRegressionCV; both pool out-of-fold predictions over all samples.
"""

import logging
import time
from collections.abc import Callable

import numpy as np
import torch
from datasets import Dataset
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nanobrain.eval.models.base import Model
from nanobrain.eval.scoring import (
    aggregate,
    balanced_accuracy_at_half,
    bootstrap_ci,
    pearson_r,
)

logger = logging.getLogger("nanobrain.eval")

FitPredict = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


# ---- feature extraction ---------------------------------------------------------------


@torch.inference_mode()
def extract_global_features(
    model: Model, dataset: Dataset, image_col: str, device: torch.device
) -> np.ndarray:
    """(N, D) one pooled embedding per subject. The model canonicalizes each nifti internally,
    so extraction is a plain per-subject loop -- no transform, no batching."""
    start = time.perf_counter()
    features = []
    for row in dataset:
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            features.append(model.global_embed(row[image_col]).float().cpu())
    X = torch.stack(features).numpy()
    logger.info(f"features {X.shape} in {time.perf_counter() - start:.1f}s")
    return X


def read_targets(dataset: Dataset, target_col: str, target_map: dict | None = None) -> np.ndarray:
    values = dataset[target_col]
    if target_map is not None:
        values = [target_map[v] for v in values]
    return np.asarray(values)


# ---- cross-validation -----------------------------------------------------------------


def repeated_oof(
    X: np.ndarray,
    y: np.ndarray,
    fit_predict: FitPredict,
    n_splits: int,
    n_repeats: int,
    seed: int,
    stratified: bool,
) -> list[np.ndarray]:
    """Out-of-fold predictions for all samples, one full array per repeat."""
    splitter_cls = StratifiedKFold if stratified else KFold
    oofs = []
    for repeat in range(n_repeats):
        splitter = splitter_cls(n_splits=n_splits, shuffle=True, random_state=seed + repeat)
        oof = np.zeros(len(y), dtype=float)
        for train_idx, test_idx in splitter.split(X, y):
            oof[test_idx] = fit_predict(X[train_idx], y[train_idx], X[test_idx])
        oofs.append(oof)
    return oofs


# ---- probes ---------------------------------------------------------------------------


def reg_probe(
    X: np.ndarray, y: np.ndarray, n_splits: int, n_repeats: int, seed: int, n_boot: int = 2000
) -> dict:
    metrics = {"mae": mean_absolute_error, "pearson_r": pearson_r}

    def fit_predict(X_train, y_train, X_test):
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
        return model.fit(X_train, y_train).predict(X_test)

    y = y.astype(float)
    oofs = repeated_oof(X, y, fit_predict, n_splits, n_repeats, seed, stratified=False)
    per_repeat = [{key: metric(y, oof) for key, metric in metrics.items()} for oof in oofs]
    summary = aggregate(per_repeat)
    summary.update(bootstrap_ci(y, np.mean(oofs, axis=0), metrics, n_boot, seed))
    return summary


def cls_probe(
    X: np.ndarray, y: np.ndarray, n_splits: int, n_repeats: int, seed: int, n_boot: int = 2000
) -> dict:
    assert set(np.unique(y)) == {0, 1}, (
        f"cls probe expects binary 0/1 labels, got {set(np.unique(y))}"
    )
    metrics = {"auroc": roc_auc_score, "balanced_accuracy": balanced_accuracy_at_half}

    def fit_predict(X_train, y_train, X_test):
        clf = LogisticRegressionCV(
            Cs=10, class_weight="balanced", scoring="balanced_accuracy", max_iter=1000
        )
        model = make_pipeline(StandardScaler(), clf).fit(X_train, y_train)
        positive = list(model.classes_).index(1)
        return model.predict_proba(X_test)[:, positive]

    oofs = repeated_oof(X, y, fit_predict, n_splits, n_repeats, seed, stratified=True)
    per_repeat = [{key: metric(y, oof) for key, metric in metrics.items()} for oof in oofs]
    summary = aggregate(per_repeat)
    summary.update(bootstrap_ci(y, np.mean(oofs, axis=0), metrics, n_boot, seed))
    return summary
