from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from datasets import Dataset as HFDataset
from sklearn.model_selection import BaseCrossValidator, GroupKFold, StratifiedGroupKFold

from evaluation.tasks.base import Kind
from evaluation.utils import build_covariates

MetricFn = Callable[..., float | np.ndarray]
PredictionPostprocessor = Callable[
    [np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    np.ndarray,
]


@dataclass
class ColumnTask:
    """Predict a single column of an HF dataset from frozen image features."""

    name: str
    kind: Kind
    data: HFDataset
    group_column: str
    metrics: Sequence[MetricFn]
    image_column: str = "image"
    target_column: str = "target"
    n_splits: int = 5
    seed: int = 0
    prediction_postprocessor: PredictionPostprocessor | None = None
    positive_label: object | None = None
    selection_metric: str = "balanced_accuracy"
    covariate_columns: tuple[str, ...] = ()
    participant_level: bool = False

    @property
    def model_selection_eligible(self) -> bool:
        return True

    def __post_init__(self):
        self.metrics = {fn.__name__: fn for fn in self.metrics}
        targets = np.asarray(self.data[self.target_column])
        present = pd.notna(targets)
        valid = present if present.ndim == 1 else present.all(axis=tuple(range(1, present.ndim)))
        if not valid.all():
            self.data = self.data.select(np.flatnonzero(valid))

    def dataset(self) -> HFDataset:
        column_mapping = {
            self.image_column: "image",
            self.target_column: "target",
        }
        dataset = self.data.select_columns(list(column_mapping)).rename_columns(column_mapping)
        return dataset

    def covariates(self) -> np.ndarray | None:
        """Nuisance covariate matrix (e.g. age, sex) for the age+sex floor, or None."""
        if not self.covariate_columns:
            return None
        return build_covariates(self.data, self.covariate_columns)

    def split(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        splitter = self._default_splitter()
        indices = np.arange(len(self.data))
        targets = np.asarray(self.data[self.target_column])
        groups = np.asarray(self.data[self.group_column])
        yield from splitter.split(indices, y=targets, groups=groups)

    def _default_splitter(self) -> BaseCrossValidator:
        kwargs = {"n_splits": self.n_splits, "shuffle": True, "random_state": self.seed}
        if self.kind == "regression":
            return GroupKFold(**kwargs)
        return StratifiedGroupKFold(**kwargs)

    def postprocess_predictions(
        self,
        y_train: np.ndarray,
        train_pred: np.ndarray,
        y_test: np.ndarray,
        test_pred: np.ndarray,
    ) -> np.ndarray:
        if self.prediction_postprocessor is None:
            return test_pred
        return self.prediction_postprocessor(y_train, train_pred, y_test, test_pred)

    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        test_idx: np.ndarray,
        y_score: np.ndarray | None = None,
    ) -> dict:
        scores = {}
        for name, metric_fn in self.metrics.items():
            value = metric_fn(
                y_true,
                y_pred,
                test_idx=test_idx,
                y_score=y_score,
                positive_label=self.positive_label,
            )
            scores[name] = float(value)
        return scores
