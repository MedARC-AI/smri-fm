"""Segmentation probe: patch-level lesion detection from dense patch features.

Feature extraction (GPU) is separated from scoring (sklearn/CPU). A patch is foreground if it
holds any lesion voxel; subject-level CV, metrics pooled over held-out patches.
"""

import logging
import time

import numpy as np
import torch
from datasets import Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nanobrain.eval.models.base import Model, Transform
from nanobrain.eval.scoring import aggregate, bootstrap_ci

logger = logging.getLogger("nanobrain.eval")

# A patch is foreground if its lesion fraction exceeds this; background patches are subsampled
# to this many per positive when fitting.
FG_THRESHOLD = 0.0
BG_PER_POS = 50


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
