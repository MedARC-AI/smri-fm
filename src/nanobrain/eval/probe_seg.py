"""Segmentation probe: patch features -> voxel-level structure detection.

The model emits localized patch features with world-mm centres. Every in-brain label voxel takes
the features of its nearest patch, so each backbone is scored on the task's own label grid whatever
its patch size. We train classifiers on subsampled voxels, then evaluate full predictions with
Dice / AP.
"""

import logging
import time
import warnings

import nibabel as nib
import numpy as np
import torch
from datasets import Dataset
from nibabel.affines import apply_affine
from scipy.spatial import cKDTree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from nanobrain.eval.models.base import Model
from nanobrain.eval.nifti import brain_mask, canonical, canonical_img
from nanobrain.eval.scoring import aggregate
from nanobrain.eval.tasks.base import SegmentationTask

logger = logging.getLogger("nanobrain.eval")

# Foreground voxels are all kept; background is capped per subject with a fixed cap
NEG_PER_SUBJECT = 10_000


# ---- embed / locate -------------------------------------------------------------------


@torch.inference_mode()
def embed_subject(
    model: Model, img: nib.Nifti1Image, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Patch features (N, D) and their world-mm centres (N, 3)."""
    with torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        features, coords = model.patch_embed(img)
    n_patches, _ = features.shape
    assert coords.shape == (n_patches, 3), (
        f"{n_patches} patch features against coords {tuple(coords.shape)}"
    )
    return features.float().numpy(), coords.float().numpy()


def voxel_targets(img: nib.Nifti1Image, seg: nib.Nifti1Image) -> tuple[np.ndarray, np.ndarray]:
    """In-brain voxels of the canonical grid: labels (V,) and world-mm coordinates (V, 3)."""
    image, affine = canonical(img), canonical_img(img).affine
    labels = canonical(seg).round().long()
    assert image.shape == labels.shape, "image and seg grids disagree"
    assert np.allclose(affine, canonical_img(seg).affine), "image and seg affines disagree"
    mask = brain_mask(image).numpy()
    return labels.numpy()[mask], apply_affine(affine, np.argwhere(mask))


def assign_patches(coords: np.ndarray, world: np.ndarray) -> np.ndarray:
    """Index of the nearest patch for each voxel, checking the patches cover the brain."""
    voxel_dist, patch = cKDTree(coords).query(world, workers=-1)
    extent = np.ptp(coords, axis=0)  # edge of the cube one patch occupies on a regular grid
    cell = float(extent.prod() / len(coords)) ** (1 / 3)
    median_dist = np.median(voxel_dist)
    assert median_dist < 2 * cell, (
        f"patches do not cover the brain: median voxel-to-patch distance {median_dist:.1f}mm "
        f"against a patch cell of {cell:.1f}mm -- suspect the world-coordinate affine"
    )
    return patch


def select_voxels(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Indices of every foreground voxel plus a capped draw of background voxels."""
    foreground = np.flatnonzero(labels > 0)
    background = np.flatnonzero(labels == 0)
    keep_bg = rng.choice(background, min(len(background), NEG_PER_SUBJECT), replace=False)
    return np.concatenate([foreground, keep_bg])


def training_subsamples(
    patches: list[tuple[np.ndarray, np.ndarray]],
    dataset: Dataset,
    task: SegmentationTask,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    subsamples = []
    for (features, coords), row in zip(patches, dataset, strict=True):
        labels, world = voxel_targets(row[task.image_col], row[task.seg_col])
        patch_of = assign_patches(coords, world)  # over the whole brain, so the guard means that
        keep = select_voxels(labels, rng)
        subsamples.append((features[patch_of[keep]], labels[keep]))
    present = {int(c) for _, y in subsamples for c in np.unique(y)}
    missing = [name for c, name in enumerate(task.class_names, 1) if c not in present]
    assert not missing, f"foreground classes absent from all segs: {missing}"
    return subsamples


# ---- fit / predict --------------------------------------------------------------------


def fit_head(x: np.ndarray, y: np.ndarray) -> Pipeline:
    clf = LogisticRegression(class_weight="balanced", max_iter=1000)
    return make_pipeline(StandardScaler(), clf).fit(x, y)


def fit_folds(
    subsamples: list[tuple[np.ndarray, np.ndarray]], n_splits: int, n_repeats: int, seed: int
) -> tuple[list[list[Pipeline]], np.ndarray]:
    """Repeated K-fold over subjects, decoupled: `heads[r][f]` is the head fit on repeat r's
    fold f, and `folds[r, i]` is the fold subject i is held out in."""
    n = len(subsamples)
    heads, folds = [], np.empty((n_repeats, n), dtype=int)
    for repeat in range(n_repeats):
        splitter = KFold(n_splits, shuffle=True, random_state=seed + repeat)
        fold_heads = []
        for fold, (train, test) in enumerate(splitter.split(range(n))):
            x = np.concatenate([subsamples[i][0] for i in train])
            y = np.concatenate([subsamples[i][1] for i in train])
            fold_heads.append(fit_head(x, y))
            folds[repeat, test] = fold
        heads.append(fold_heads)
    return heads, folds


def predict_probs(head: Pipeline, x: np.ndarray, n_classes: int) -> np.ndarray:
    """(N, K+1) probabilities. A fold that never saw a class emits no column for it, so that
    column stays zero and the class is simply never predicted."""
    probs = np.zeros((len(x), n_classes + 1), dtype=np.float32)
    probs[:, head.classes_] = head.predict_proba(x)
    return probs


# ---- score ----------------------------------------------------------------------------


def dice_score(pred: np.ndarray, truth: np.ndarray) -> float:
    denom = int(pred.sum()) + int(truth.sum())
    return 2 * int(np.logical_and(pred, truth).sum()) / denom if denom else 1.0


def score_prediction(y_true: np.ndarray, probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per foreground class, (dice, ap) for one prediction. A class with no ground-truth voxels
    scores specificity in Dice (1 iff no false positive) and NaN for AP."""
    n_classes = probs.shape[1] - 1
    pred = probs.argmax(axis=1)
    dice, ap = np.empty(n_classes), np.full(n_classes, np.nan)
    for c in range(1, n_classes + 1):
        truth = y_true == c
        if truth.any():
            dice[c - 1] = dice_score(pred == c, truth)
            ap[c - 1] = average_precision_score(truth, probs[:, c])
        else:
            dice[c - 1] = float(not (pred == c).any())
    return dice, ap


def score_dataset(
    patches: list[tuple[np.ndarray, np.ndarray]],
    dataset: Dataset,
    task: SegmentationTask,
    heads: list[list[Pipeline]],
    folds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict every in-brain voxel under the heads that held its subject out, and score per class.

    Nearest assignment makes the prediction constant within a patch, so the head runs over the N
    patches and the result is gathered out to voxels.
    """
    n_classes = len(task.class_names)
    n_repeats = len(heads)
    dice = np.full((n_classes, len(dataset), n_repeats), np.nan)
    ap = np.full((n_classes, len(dataset), n_repeats), np.nan)
    for i, ((features, coords), row) in enumerate(zip(patches, dataset, strict=True)):
        labels, world = voxel_targets(row[task.image_col], row[task.seg_col])
        patch_of = assign_patches(coords, world)
        for r in range(n_repeats):
            probs = predict_probs(heads[r][folds[r, i]], features, n_classes)
            dice[:, i, r], ap[:, i, r] = score_prediction(labels, probs[patch_of])
    return dice, ap


def summarize(
    dice: np.ndarray, ap: np.ndarray, class_names: tuple[str, ...], n_boot: int, seed: int
) -> dict:
    """Per-class and macro Dice / voxel-AP: point estimate over repeats plus a subject bootstrap."""
    _, _, n_repeats = dice.shape
    metrics: dict[str, np.ndarray] = {}
    for family, arr in (("dice", dice), ("voxel_ap", ap)):
        for c, name in enumerate(class_names):
            metrics[f"{family}_{name}"] = arr[c]
        metrics[family] = nanmean(arr, axis=0)  # macro over classes

    per_repeat = [
        {key: float(nanmean(mat[:, r])) for key, mat in metrics.items()} for r in range(n_repeats)
    ]
    summary = aggregate(per_repeat)
    for key, mat in metrics.items():
        low, high = bootstrap_subject(nanmean(mat, axis=1), n_boot, seed)
        summary[f"{key}_ci_low"], summary[f"{key}_ci_high"] = low, high
    return summary


def bootstrap_subject(
    per_subject: np.ndarray, n_boot: int, seed: int, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile CI on the subject-mean, resampling subjects with replacement."""
    rng = np.random.default_rng(seed)
    n = len(per_subject)
    samples = [nanmean(per_subject[rng.integers(0, n, size=n)]) for _ in range(n_boot)]
    low, high = np.nanpercentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(low), float(high)


def nanmean(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    # Subjects with no ground-truth voxels of a class have no AP, so some slices are all-NaN.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
        return np.nanmean(values, axis=axis)


# ---- probe ----------------------------------------------------------------------------


def seg_probe(
    model: Model,
    task: SegmentationTask,
    *,
    device: torch.device,
    n_splits: int,
    n_repeats: int,
    seed: int,
    n_boot: int = 2000,
) -> dict:
    """Voxel-level detection over K foreground classes: subject-level repeated CV, scored by
    per-subject Dice and average precision."""
    start = time.perf_counter()
    dataset = task.dataset_fn()
    logger.info(f"dataset: {len(dataset)} samples")
    patches = [embed_subject(model, row[task.image_col], device) for row in dataset]
    subsamples = training_subsamples(patches, dataset, task, seed)
    heads, folds = fit_folds(subsamples, n_splits, n_repeats, seed)
    dice, ap = score_dataset(patches, dataset, task, heads, folds)
    logger.info(f"seg probe over {len(dataset)} subjects in {time.perf_counter() - start:.1f}s")
    return summarize(dice, ap, task.class_names, n_boot, seed)
