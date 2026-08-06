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
