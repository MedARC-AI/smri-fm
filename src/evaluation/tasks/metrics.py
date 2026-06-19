import numpy as np
from sklearn import metrics as skm


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    # Multi-output targets (e.g. the 101 SynthSeg regional volumes) must keep their
    # per-output structure. Flattening pools outputs that span orders of magnitude,
    # which makes a single r2/mae meaningless. Here r2 is averaged over outputs and
    # errors are reported relative to each output's magnitude so heterogeneously
    # scaled regions are comparable.
    if y_true.ndim == 2 and y_true.shape[1] > 1:
        scale = np.maximum(np.abs(y_true).mean(axis=0), 1e-8)
        mae = np.abs(y_pred - y_true).mean(axis=0)
        rmse = np.sqrt(((y_pred - y_true) ** 2).mean(axis=0))
        return {
            "r2": float(skm.r2_score(y_true, y_pred)),
            "rel_mae": float((mae / scale).mean()),
            "rel_rmse": float((rmse / scale).mean()),
        }

    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    residuals = y_pred - y_true
    return {
        "mae": float(np.abs(residuals).mean()),
        "rmse": float(np.sqrt((residuals**2).mean())),
        "r2": float(skm.r2_score(y_true, y_pred)),
    }


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(skm.accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(skm.balanced_accuracy_score(y_true, y_pred)),
    }
