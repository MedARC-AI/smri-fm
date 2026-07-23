import nibabel as nib
import numpy as np
import pytest
import torch
from datasets import Dataset, Features, Nifti

from nanobrain.eval import probe as probe_module
from nanobrain.eval.models import create_model
from sklearn.metrics import roc_auc_score

from nanobrain.eval.probe import (
    aggregate,
    bootstrap_ci,
    cls_probe,
    extract_global_features,
    reg_probe,
    repeated_oof,
    seg_probe,
)


def make_nifti(seed: int, shape=(24, 28, 26)) -> dict:
    rng = np.random.default_rng(seed)
    data = rng.random(shape, dtype=np.float32)
    img = nib.Nifti1Image(data, affine=np.eye(4))
    return {"path": None, "bytes": img.to_bytes()}


def test_reg_probe_recovers_linear_signal():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, 8))
    y = X @ rng.standard_normal(8) + 0.1 * rng.standard_normal(60)
    scores = reg_probe(X, y, n_splits=5, n_repeats=2, seed=0)
    assert set(scores) == {
        "mae", "mae_std", "mae_ci_low", "mae_ci_high",
        "pearson_r", "pearson_r_std", "pearson_r_ci_low", "pearson_r_ci_high",
    }  # fmt: skip
    assert scores["pearson_r"] > 0.9
    assert scores["mae"] < y.std()
    assert scores["pearson_r_ci_low"] <= scores["pearson_r"] <= scores["pearson_r_ci_high"]


def test_cls_probe_separable():
    rng = np.random.default_rng(0)
    X = np.concatenate([rng.standard_normal((30, 8)) - 1, rng.standard_normal((30, 8)) + 1])
    y = np.array([0] * 30 + [1] * 30)
    scores = cls_probe(X, y, n_splits=5, n_repeats=2, seed=0)
    assert scores["auroc"] > 0.9
    assert 0.0 <= scores["balanced_accuracy"] <= 1.0


def test_repeated_oof_covers_each_sample_once():
    # X encodes each row's index; the fake head echoes it, so a correct out-of-fold
    # scatter must reconstruct arange(n) -- every sample predicted exactly once, in place.
    n = 12
    X = np.arange(n, dtype=float).reshape(n, 1)
    y = np.zeros(n)
    oofs = repeated_oof(
        X, y, lambda _Xtr, _ytr, X_te: X_te[:, 0], n_splits=4, n_repeats=2, seed=0, stratified=False
    )
    assert len(oofs) == 2
    for oof in oofs:
        assert np.array_equal(oof, np.arange(n))


def test_reg_probe_deterministic():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((40, 6))
    y = X @ rng.standard_normal(6)
    assert reg_probe(X, y, 5, 2, seed=7) == reg_probe(X, y, 5, 2, seed=7)


def test_cls_probe_positive_column_with_minority_class():
    # class 1 is the minority; if the wrong proba column were scored, AUROC would be < 0.5.
    rng = np.random.default_rng(0)
    X = np.concatenate([rng.standard_normal((50, 6)) - 1.5, rng.standard_normal((10, 6)) + 1.5])
    y = np.array([0] * 50 + [1] * 10)
    assert cls_probe(X, y, n_splits=5, n_repeats=2, seed=0)["auroc"] > 0.9


def test_seg_probe_requires_foreground():
    features = [np.zeros((10, 3)) for _ in range(4)]
    fractions = [np.zeros(10) for _ in range(4)]
    with pytest.raises(AssertionError, match="foreground"):
        seg_probe(features, fractions, n_splits=2, n_repeats=1, seed=0)


def test_aggregate_rejects_non_finite():
    with pytest.raises(AssertionError, match="non-finite"):
        aggregate([{"metric": float("nan")}])


def test_cls_probe_rejects_non_binary():
    X = np.zeros((10, 3))
    y = np.array([0, 1, 2] * 3 + [0])
    try:
        cls_probe(X, y, n_splits=2, n_repeats=1, seed=0)
        raise AssertionError("expected non-binary labels to be rejected")
    except AssertionError as err:
        assert "binary" in str(err)


def test_random_features_contract():
    model, transform = create_model("random_features", size=32, patch=8, dim=64)
    img = nib.Nifti1Image(
        np.random.default_rng(0).random((20, 24, 22), dtype=np.float32), np.eye(4)
    )
    seg_data = np.zeros((20, 24, 22), dtype=np.float32)
    seg_data[8:12, 10:14, 9:13] = 1.0
    seg = nib.Nifti1Image(seg_data, np.eye(4))

    sample = transform(img, seg)
    batch = {"image": sample["image"].unsqueeze(0)}
    assert model.global_embed(batch).shape == (1, 64)

    patches = model.patch_embed(batch)
    labels = model.patchify_labels(sample["seg"])
    n_patches = (32 // 8) ** 3
    assert patches.shape == (1, n_patches, 64)
    assert labels.shape == (n_patches,)  # aligned with patch_embed by construction


def test_extract_global_features():
    dataset = Dataset.from_dict(
        {"image": [make_nifti(i) for i in range(6)]}, features=Features({"image": Nifti()})
    )
    model, transform = create_model("random_features", size=32, patch=8, dim=64)
    X = extract_global_features(
        model, transform, dataset, "image", torch.device("cpu"), batch_size=4, num_workers=0
    )
    assert X.shape == (6, 64)
    assert np.isfinite(X).all()


def test_seg_probe_detects_separable_patches():
    rng = np.random.default_rng(0)
    features, fractions = [], []
    for _ in range(8):
        feats = rng.standard_normal((50, 4))
        frac = np.zeros(50)
        frac[:5] = 1.0  # 5 foreground patches per subject
        feats[:5] += 3.0  # made linearly separable
        features.append(feats)
        fractions.append(frac)
    scores = seg_probe(features, fractions, n_splits=4, n_repeats=2, seed=0)
    assert set(scores) == {
        "detection_ap", "detection_ap_std", "detection_ap_ci_low", "detection_ap_ci_high",
        "patch_auroc", "patch_auroc_std", "patch_auroc_ci_low", "patch_auroc_ci_high",
    }  # fmt: skip
    assert scores["detection_ap"] > 0.8
    assert scores["patch_auroc_ci_low"] <= scores["patch_auroc"] <= scores["patch_auroc_ci_high"]


def test_random_features_patchify_matches_grid():
    # A localized mask must produce some foreground and some background patches.
    model, transform = create_model("random_features", size=32, patch=8, dim=16)
    img = nib.Nifti1Image(np.ones((20, 24, 22), dtype=np.float32), np.eye(4))
    seg_data = np.zeros((20, 24, 22), dtype=np.float32)
    seg_data[8:12, 10:14, 9:13] = 1.0
    sample = transform(img, nib.Nifti1Image(seg_data, np.eye(4)))
    fractions = model.patchify_labels(sample["seg"])
    assert fractions.shape == ((32 // 8) ** 3,)
    assert (fractions > 0).any() and (fractions == 0).any()


def test_bootstrap_ci_brackets_point_and_is_deterministic():
    rng = np.random.default_rng(0)
    y_true = np.array([0] * 50 + [1] * 50)
    y_score = np.clip(y_true + 0.3 * rng.standard_normal(100), 0, 1)
    metrics = {"auroc": roc_auc_score}
    ci = bootstrap_ci(y_true, y_score, metrics, n_boot=500, seed=0)
    point = roc_auc_score(y_true, y_score)
    assert ci["auroc_ci_low"] < point < ci["auroc_ci_high"]
    assert ci == bootstrap_ci(y_true, y_score, metrics, n_boot=500, seed=0)


def test_bootstrap_ci_resamples_by_group():
    # Two subjects, one all-positive and one all-negative: a per-row bootstrap could draw
    # a single class, but grouped resampling keeps each subject's rows together, so any
    # non-degenerate draw (both subjects picked) has both classes present.
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_score = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.3])
    groups = np.array([0, 0, 0, 1, 1, 1])
    ci = bootstrap_ci(y_true, y_score, {"auroc": roc_auc_score}, n_boot=200, seed=1, groups=groups)
    # Draws that pick the same subject twice are single-class and dropped; the rest score 1.0.
    assert ci["auroc_ci_low"] == 1.0 and ci["auroc_ci_high"] == 1.0


def test_read_targets_remaps():
    dataset = Dataset.from_dict({"dx": ["CN", "AD", "CN"]})
    y = probe_module.read_targets(dataset, "dx", target_map={"CN": 0, "AD": 1})
    assert list(y) == [0, 1, 0]
