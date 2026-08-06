import nibabel as nib
import numpy as np
import pytest
import torch
from datasets import Dataset, Features, Nifti, Value

from nanobrain.eval.models import create_model
from nanobrain.eval.probe_global import (
    CLS_METRICS,
    REG_METRICS,
    extract_features,
    fit_logistic,
    fit_ridge,
    read_targets,
    repeated_oof,
    summarize,
    cls_probe,
    reg_probe,
)
from nanobrain.eval.tasks.base import ClassificationTask, RegressionTask

CPU = torch.device("cpu")


def make_nifti(seed: int, shape=(24, 28, 26)) -> dict:
    rng = np.random.default_rng(seed)
    data = rng.random(shape, dtype=np.float32)
    img = nib.Nifti1Image(data, affine=np.eye(4))
    return {"path": None, "bytes": img.to_bytes()}


def make_dataset(n: int, target_col: str, targets: list) -> Dataset:
    return Dataset.from_dict(
        {"image": [make_nifti(i) for i in range(n)], target_col: targets},
        features=Features({"image": Nifti(), target_col: Value("int32")}),
    )


def test_ridge_head_recovers_linear_signal():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, 8))
    y = X @ rng.standard_normal(8) + 0.1 * rng.standard_normal(60)
    oofs = repeated_oof(X, y, fit_ridge, n_splits=5, n_repeats=2, seed=0, stratified=False)
    scores = summarize(y, oofs, REG_METRICS, n_boot=2000, seed=0)
    assert set(scores) == {
        "mae", "mae_std", "mae_ci_low", "mae_ci_high",
        "pearson_r", "pearson_r_std", "pearson_r_ci_low", "pearson_r_ci_high",
    }  # fmt: skip
    assert scores["pearson_r"] > 0.9
    assert scores["mae"] < y.std()
    assert scores["pearson_r_ci_low"] <= scores["pearson_r"] <= scores["pearson_r_ci_high"]


def test_logistic_head_separable():
    rng = np.random.default_rng(0)
    X = np.concatenate([rng.standard_normal((30, 8)) - 1, rng.standard_normal((30, 8)) + 1])
    y = np.array([0] * 30 + [1] * 30)
    oofs = repeated_oof(X, y, fit_logistic, n_splits=5, n_repeats=2, seed=0, stratified=True)
    scores = summarize(y, oofs, CLS_METRICS, n_boot=2000, seed=0)
    assert scores["auroc"] > 0.9
    assert 0.0 <= scores["balanced_accuracy"] <= 1.0


def test_logistic_head_positive_column_with_minority_class():
    # class 1 is the minority; if the wrong proba column were scored, AUROC would be < 0.5.
    rng = np.random.default_rng(0)
    X = np.concatenate([rng.standard_normal((50, 6)) - 1.5, rng.standard_normal((10, 6)) + 1.5])
    y = np.array([0] * 50 + [1] * 10)
    oofs = repeated_oof(X, y, fit_logistic, n_splits=5, n_repeats=2, seed=0, stratified=True)
    assert summarize(y, oofs, CLS_METRICS, n_boot=200, seed=0)["auroc"] > 0.9


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


def test_summarize_deterministic():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((40, 6))
    y = X @ rng.standard_normal(6)
    oofs = repeated_oof(X, y, fit_ridge, 5, 2, seed=7, stratified=False)
    assert summarize(y, oofs, REG_METRICS, 2000, 7) == summarize(y, oofs, REG_METRICS, 2000, 7)


def test_extract_features():
    dataset = Dataset.from_dict(
        {"image": [make_nifti(i) for i in range(6)]}, features=Features({"image": Nifti()})
    )
    model = create_model("random_features", size=32, patch=8, dim=64)
    X = extract_features(model, dataset, "image", CPU)
    assert X.shape == (6, 64)
    assert np.isfinite(X).all()


def test_read_targets_remaps():
    dataset = Dataset.from_dict({"dx": ["CN", "AD", "CN"]})
    y = read_targets(dataset, "dx", target_map={"CN": 0, "AD": 1})
    assert list(y) == [0, 1, 0]


def test_reg_probe_matches_its_parts():
    # the seam main.py calls: reg_probe must equal extract + read + oof + summarize by hand.
    dataset = make_dataset(12, "age", list(range(20, 32)))
    task = RegressionTask(name="fake", dataset_fn=lambda: dataset, target_col="age")
    model = create_model("random_features", size=32, patch=8, dim=64)
    scores = reg_probe(model, task, dataset, CPU, n_splits=3, n_repeats=1, seed=0, n_boot=50)

    X = extract_features(model, dataset, task.image_col, CPU)
    y = read_targets(dataset, task.target_col).astype(float)
    oofs = repeated_oof(X, y, fit_ridge, 3, 1, seed=0, stratified=False)
    assert scores == summarize(y, oofs, REG_METRICS, 50, 0)
    assert np.isfinite(scores["mae"])


def test_cls_probe_rejects_non_binary_before_extracting():
    dataset = make_dataset(6, "label", [0, 1, 2, 0, 1, 2])
    task = ClassificationTask(name="fake", dataset_fn=lambda: dataset, target_col="label")
    with pytest.raises(AssertionError, match="binary"):
        cls_probe(None, task, dataset, CPU, n_splits=2, n_repeats=1, seed=0)
