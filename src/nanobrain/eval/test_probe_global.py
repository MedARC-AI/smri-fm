import nibabel as nib
import numpy as np
import torch
from datasets import Dataset, Features, Nifti

from nanobrain.eval.probe_global import (
    cls_probe,
    extract_global_features,
    read_targets,
    reg_probe,
    repeated_oof,
)
from nanobrain.eval.models import create_model


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


def test_cls_probe_rejects_non_binary():
    X = np.zeros((10, 3))
    y = np.array([0, 1, 2] * 3 + [0])
    try:
        cls_probe(X, y, n_splits=2, n_repeats=1, seed=0)
        raise AssertionError("expected non-binary labels to be rejected")
    except AssertionError as err:
        assert "binary" in str(err)


def test_extract_global_features():
    dataset = Dataset.from_dict(
        {"image": [make_nifti(i) for i in range(6)]}, features=Features({"image": Nifti()})
    )
    model = create_model("random_features", size=32, patch=8, dim=64)
    X = extract_global_features(model, dataset, "image", torch.device("cpu"))
    assert X.shape == (6, 64)
    assert np.isfinite(X).all()


def test_read_targets_remaps():
    dataset = Dataset.from_dict({"dx": ["CN", "AD", "CN"]})
    y = read_targets(dataset, "dx", target_map={"CN": 0, "AD": 1})
    assert list(y) == [0, 1, 0]
