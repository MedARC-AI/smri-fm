"""Shared scoring for the frozen-feature probes: repeat aggregation, bootstrap CIs, metrics.

Each probe pools out-of-fold predictions per repeat and averages the metric across repeats;
the reported std is the spread across repeats -- a rough stability signal, not a confidence
interval (repeated-CV folds share data, so it understates true sampling variance). The CI is
a separate subject-level bootstrap.
"""

import logging
from collections.abc import Callable

import numpy as np
from sklearn.metrics import balanced_accuracy_score

logger = logging.getLogger("nanobrain.eval")

Metric = Callable[[np.ndarray, np.ndarray], float]


def aggregate(per_repeat: list[dict[str, float]]) -> dict[str, float]:
    summary = {}
    for key in per_repeat[0]:
        values = np.array([scores[key] for scores in per_repeat])
        summary[key] = float(values.mean())
        summary[f"{key}_std"] = float(values.std())
    # A non-finite metric means a degenerate fit/fold; surface it instead of logging nan.
    assert np.isfinite(list(summary.values())).all(), f"non-finite metric: {summary}"
    return summary


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metrics: dict[str, Metric],
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Percentile CI per metric, resampling subjects with replacement.

    Resamples with a single unique label, where a metric is undefined, are skipped and counted.
    """
    n_subjects = len(y_true)
    rng = np.random.default_rng(seed)

    samples: dict[str, list[float]] = {key: [] for key in metrics}
    n_dropped = 0
    for _ in range(n_boot):
        rows = rng.integers(0, n_subjects, size=n_subjects)
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


def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def balanced_accuracy_at_half(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return balanced_accuracy_score(y_true, y_score > 0.5)
