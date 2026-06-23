import numpy as np
from scipy import stats
from sklearn import metrics as skm


def accuracy(y_true: np.ndarray, y_pred: np.ndarray, **_) -> float:
    return float(skm.accuracy_score(y_true, y_pred))


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, **_) -> float:
    return float(skm.balanced_accuracy_score(y_true, y_pred))


def auroc(
    y_true: np.ndarray,
    y_pred: np.ndarray | None = None,
    *,
    y_score: np.ndarray | None = None,
    positive_label,
    **_,
) -> float:
    y_score = y_pred if y_score is None else y_score
    binary_true = np.asarray(y_true) == positive_label
    return float(skm.roc_auc_score(binary_true, y_score))


def auprc(
    y_true: np.ndarray,
    y_pred: np.ndarray | None = None,
    *,
    y_score: np.ndarray | None = None,
    positive_label,
    **_,
) -> float:
    y_score = y_pred if y_score is None else y_score
    binary_true = np.asarray(y_true) == positive_label
    return float(skm.average_precision_score(binary_true, y_score))


def mae(y_true: np.ndarray, y_pred: np.ndarray, **_) -> float:
    return float(skm.mean_absolute_error(y_true, y_pred))


def rmse(y_true: np.ndarray, y_pred: np.ndarray, **_) -> float:
    return float(np.sqrt(skm.mean_squared_error(y_true, y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray, **_) -> float:
    return float(skm.r2_score(y_true, y_pred, multioutput="uniform_average"))


def pearson_r(y_true: np.ndarray, y_pred: np.ndarray, **_) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.ndim == 2:
        return float(np.nanmean([pearson_r(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]))
    y_true, y_pred = y_true.reshape(-1), y_pred.reshape(-1)
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def spearman_r(y_true: np.ndarray, y_pred: np.ndarray, **_) -> float:
    return float(stats.spearmanr(np.asarray(y_true).reshape(-1), np.asarray(y_pred).reshape(-1)).statistic)


def multiclass_auroc(
    y_true: np.ndarray,
    y_pred: np.ndarray | None = None,
    *,
    y_score: np.ndarray | None = None,
    **_,
) -> float:
    y_score = y_pred if y_score is None else y_score
    return float(skm.roc_auc_score(y_true, y_score, multi_class="ovr", average="macro"))


def cohen_kappa(y_true: np.ndarray, y_pred: np.ndarray, **_) -> float:
    return float(skm.cohen_kappa_score(y_true, y_pred, weights="quadratic"))


def independent_t_statistic(a: np.ndarray, b: np.ndarray) -> float:
    return float(stats.ttest_ind(a, b).statistic)



def age_bias_correct_predictions(
    y_train: np.ndarray,
    train_pred: np.ndarray,
    y_test: np.ndarray,
    test_pred: np.ndarray,
) -> np.ndarray:
    slope, intercept = np.polyfit(
        np.asarray(y_train).reshape(-1),
        np.asarray(train_pred).reshape(-1),
        deg=1,
    )
    if np.isclose(slope, 0):
        return test_pred
    return (np.asarray(test_pred) - intercept) / slope
