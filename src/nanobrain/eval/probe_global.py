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
from nanobrain.eval.tasks.base import ClassificationTask, RegressionTask

logger = logging.getLogger("nanobrain.eval")

FitPredict = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]

REG_METRICS = {"mae": mean_absolute_error, "pearson_r": pearson_r}
CLS_METRICS = {"auroc": roc_auc_score, "balanced_accuracy": balanced_accuracy_at_half}


# ---- embed / targets ------------------------------------------------------------------


@torch.inference_mode()
def _extract_features(
    model: Model, dataset: Dataset, image_col: str, device: torch.device
) -> np.ndarray:
    """(N, D) one pooled embedding per subject. The model canonicalizes each nifti internally,
    so extraction is a plain per-subject loop -- no transform, no batching."""
    start = time.perf_counter()
    features = []
    for ii, row in enumerate(dataset):
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            features.append(model.global_embed(row[image_col]).float().cpu())
        if (ii + 1) % 10 == 0:
            rate = (time.perf_counter() - start) / (ii + 1)
            logger.info(f"embedded {ii + 1}/{len(dataset)} at {rate:.2f}s/volume")
    X = torch.stack(features).numpy()
    logger.info(f"features {X.shape} in {time.perf_counter() - start:.1f}s")
    return X


def _read_targets(dataset: Dataset, target_col: str, target_map: dict | None = None) -> np.ndarray:
    values = dataset[target_col]
    if target_map is not None:
        values = [target_map[v] for v in values]
    return np.asarray(values)


# ---- fit / predict --------------------------------------------------------------------


def _fit_ridge(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    head = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
    return head.fit(X_train, y_train).predict(X_test)


def _fit_logistic(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """Positive-class probability. Indexes `classes_` rather than assuming column 1, which would
    silently score the wrong class when the label order differs."""
    clf = LogisticRegressionCV(
        Cs=10,
        class_weight="balanced",
        scoring="balanced_accuracy",
        max_iter=1000,
        l1_ratios=(0,),
        use_legacy_attributes=False,
    )
    head = make_pipeline(StandardScaler(), clf).fit(X_train, y_train)
    positive = list(head.classes_).index(1)
    return head.predict_proba(X_test)[:, positive]


def _repeated_oof(
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


# ---- score ----------------------------------------------------------------------------


def _summarize(
    y: np.ndarray, oofs: list[np.ndarray], metrics: dict, n_boot: int, seed: int
) -> dict:
    """Point estimate over repeats plus a bootstrap CI on the repeat-averaged predictions."""
    per_repeat = [{key: metric(y, oof) for key, metric in metrics.items()} for oof in oofs]
    summary = aggregate(per_repeat)
    summary.update(bootstrap_ci(y, np.mean(oofs, axis=0), metrics, n_boot, seed))
    return summary


# ---- probes ---------------------------------------------------------------------------


def reg_probe(
    model: Model,
    task: RegressionTask,
    dataset: Dataset,
    device: torch.device,
    n_splits: int,
    n_repeats: int,
    seed: int,
    n_boot: int = 2000,
) -> dict:
    """Scalar regression off the pooled embedding, scored by MAE and Pearson r."""
    start = time.perf_counter()
    X = _extract_features(model, dataset, task.image_col, device)
    y = _read_targets(dataset, task.target_col).astype(float)
    oofs = _repeated_oof(X, y, _fit_ridge, n_splits, n_repeats, seed, stratified=False)
    logger.info(f"reg probe over {len(dataset)} subjects in {time.perf_counter() - start:.1f}s")
    return _summarize(y, oofs, REG_METRICS, n_boot, seed)


def cls_probe(
    model: Model,
    task: ClassificationTask,
    dataset: Dataset,
    device: torch.device,
    n_splits: int,
    n_repeats: int,
    seed: int,
    n_boot: int = 2000,
) -> dict:
    """Binary classification off the pooled embedding, scored by AUROC and balanced accuracy."""
    start = time.perf_counter()
    y = _read_targets(dataset, task.target_col, task.target_map)
    assert set(np.unique(y)) == {0, 1}, (
        f"cls probe expects binary 0/1 labels, got {set(np.unique(y))}"
    )
    X = _extract_features(model, dataset, task.image_col, device)
    oofs = _repeated_oof(X, y, _fit_logistic, n_splits, n_repeats, seed, stratified=True)
    logger.info(f"cls probe over {len(dataset)} subjects in {time.perf_counter() - start:.1f}s")
    return _summarize(y, oofs, CLS_METRICS, n_boot, seed)
