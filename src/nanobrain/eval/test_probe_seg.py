import numpy as np
import pytest

from nanobrain.eval.probe_seg import seg_probe


def test_seg_probe_requires_foreground():
    features = [np.zeros((10, 3)) for _ in range(4)]
    fractions = [np.zeros(10) for _ in range(4)]
    with pytest.raises(AssertionError, match="foreground"):
        seg_probe(features, fractions, n_splits=2, n_repeats=1, seed=0)


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
