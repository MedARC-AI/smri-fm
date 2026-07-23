"""Frozen-feature sklearn probes.

Feature extraction (GPU) is separated from scoring (sklearn/CPU). Each probe scores by
pooling out-of-fold predictions over all samples within a repeat, then averaging the
metric across repeats -- so rank metrics like AUROC get all N test points. The reported
std is the spread across repeats; it understates true sampling variance (repeated-CV folds
share data), so read it as a rough stability signal, not a confidence interval.
"""

import logging
import time
from collections.abc import Callable

import numpy as np
import torch
from datasets import Dataset
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, RidgeCV
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    mean_absolute_error,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from nanobrain.eval.models.base import Model, Transform

logger = logging.getLogger("nanobrain.eval")

FitPredict = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]

# Segmentation probe: a patch is foreground if it holds any lesion voxel; background patches
# are subsampled to this many per positive when fitting.
FG_THRESHOLD = 0.0
BG_PER_POS = 50


# ---- feature extraction ---------------------------------------------------------------


class _TransformDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: Dataset, transform: Transform, image_col: str):
        self.dataset = dataset
        self.transform = transform
        self.image_col = image_col

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, i: int) -> dict:
        return self.transform(self.dataset[i][self.image_col])


@torch.inference_mode()
def extract_global_features(
    model: Model,
    transform: Transform,
    dataset: Dataset,
    image_col: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> np.ndarray:
    loader = DataLoader(
        _TransformDataset(dataset, transform, image_col),
        batch_size=batch_size,
        num_workers=num_workers,
    )
    start = time.perf_counter()
    features = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            features.append(model.global_embed(batch).float().cpu())
    X = torch.cat(features).numpy()
    logger.info(f"features {X.shape} in {time.perf_counter() - start:.1f}s")
    return X


def read_targets(dataset: Dataset, target_col: str, target_map: dict | None = None) -> np.ndarray:
    values = dataset[target_col]
    if target_map is not None:
        values = [target_map[v] for v in values]
    return np.asarray(values)


@torch.inference_mode()
def extract_patch_features(
    model: Model,
    transform: Transform,
    dataset: Dataset,
    image_col: str,
    seg_col: str,
    device: torch.device,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Per subject: (patch features (N, D), per-patch foreground fraction (N,)).

    One subject at a time (n is tiny for the seg tasks); the model owns the patch grid,
    so features[i] and fractions[i] are index-aligned by construction.
    """
    start = time.perf_counter()
    features, fractions = [], []
    for row in dataset:
        sample = transform(row[image_col], row[seg_col])
        seg_grid = sample.pop("seg")
        batch = {k: v.unsqueeze(0).to(device) for k, v in sample.items()}
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            feat = model.patch_embed(batch)[0].float().cpu().numpy()
        frac = model.patchify_labels(seg_grid).cpu().numpy()
        # The alignment invariant: patch tokens and patch labels share one grid.
        assert len(feat) == len(frac), f"patch/label grid mismatch: {len(feat)} vs {len(frac)}"
        features.append(feat)
        fractions.append(frac)
    n_patches = sum(len(f) for f in features)
    logger.info(
        f"patches {n_patches} over {len(features)} subjects in {time.perf_counter() - start:.1f}s"
    )
    return features, fractions


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


def aggregate(per_repeat: list[dict[str, float]]) -> dict[str, float]:
    summary = {}
    for key in per_repeat[0]:
        values = np.array([scores[key] for scores in per_repeat])
        summary[key] = float(values.mean())
        summary[f"{key}_std"] = float(values.std())
    # A non-finite metric means a degenerate fit/fold; surface it instead of logging nan.
    assert np.isfinite(list(summary.values())).all(), f"non-finite metric: {summary}"
    return summary


Metric = Callable[[np.ndarray, np.ndarray], float]


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metrics: dict[str, Metric],
    n_boot: int,
    seed: int,
    groups: np.ndarray | None = None,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Percentile CI per metric, resampling subjects with replacement.

    `groups` gives a subject id per row so a subject's rows resample together (seg has many
    patch rows per subject); None means one row per subject. Resamples with a single unique
    label, where a metric is undefined, are skipped and counted.
    """
    if groups is None:
        groups = np.arange(len(y_true))
    unique = np.unique(groups)
    rows_by_group = [np.flatnonzero(groups == g) for g in unique]
    rng = np.random.default_rng(seed)

    samples: dict[str, list[float]] = {key: [] for key in metrics}
    n_dropped = 0
    for _ in range(n_boot):
        pick = rng.integers(0, len(unique), size=len(unique))
        rows = np.concatenate([rows_by_group[i] for i in pick])
        yt, ys = y_true[rows], y_score[rows]
        if len(np.unique(yt)) < 2:
            n_dropped += 1
            continue
        for key, metric in metrics.items():
            samples[key].append(metric(yt, ys))

    if n_dropped:
        logger.info(f"bootstrap dropped {n_dropped} single-label resamples")

    ci = {}
    for key, values in samples.items():
        low, high = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        ci[f"{key}_ci_low"] = float(low)
        ci[f"{key}_ci_high"] = float(high)
    return ci


def balanced_accuracy_at_half(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return balanced_accuracy_score(y_true, y_score > 0.5)


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


def seg_probe(
    features: list[np.ndarray],
    fractions: list[np.ndarray],
    n_splits: int,
    n_repeats: int,
    seed: int,
    n_boot: int = 2000,
) -> dict:
    """Patch-level lesion detection. Subject-level CV; metrics pooled over held-out patches."""
    labels = [(frac > FG_THRESHOLD).astype(int) for frac in fractions]
    y_true = np.concatenate(labels)
    # No positive patches means every mask vanished (e.g. resized below the patch grid).
    assert set(np.unique(y_true)) == {0, 1}, "seg labels have no foreground patches"
    n_subjects = len(features)
    metrics = {"detection_ap": average_precision_score, "patch_auroc": roc_auc_score}

    score_sum = [np.zeros(len(feat)) for feat in features]
    per_repeat = []
    for repeat in range(n_repeats):
        rng = np.random.default_rng(seed + repeat)
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed + repeat)
        oof_scores: list[np.ndarray | None] = [None] * n_subjects
        for train_idx, test_idx in splitter.split(np.arange(n_subjects)):
            X_train = np.concatenate([features[i] for i in train_idx])
            y_train = np.concatenate([labels[i] for i in train_idx])
            X_train, y_train = _subsample_background(X_train, y_train, rng)
            clf = make_pipeline(
                StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)
            ).fit(X_train, y_train)
            positive = list(clf.classes_).index(1)
            for i in test_idx:
                oof_scores[i] = clf.predict_proba(features[i])[:, positive]
        for i in range(n_subjects):
            score_sum[i] = score_sum[i] + oof_scores[i]
        scores = np.concatenate(oof_scores)
        per_repeat.append({key: metric(y_true, scores) for key, metric in metrics.items()})
    summary = aggregate(per_repeat)

    y_score = np.concatenate([total / n_repeats for total in score_sum])
    groups = np.concatenate([np.full(len(feat), i) for i, feat in enumerate(features)])
    summary.update(bootstrap_ci(y_true, y_score, metrics, n_boot, seed, groups=groups))
    return summary


def _subsample_background(
    X: np.ndarray, y: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    n_keep = min(len(negative), max(len(positive) * BG_PER_POS, 1))
    keep = np.concatenate([positive, rng.choice(negative, size=n_keep, replace=False)])
    return X[keep], y[keep]


def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.corrcoef(y_true, y_pred)[0, 1])
