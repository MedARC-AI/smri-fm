import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from nanobrain.eval.scoring import aggregate, bootstrap_ci


def test_aggregate_rejects_non_finite():
    with pytest.raises(AssertionError, match="non-finite"):
        aggregate([{"metric": float("nan")}])


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
