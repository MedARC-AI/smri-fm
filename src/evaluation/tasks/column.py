from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from datasets import Dataset as HFDataset
from sklearn.model_selection import (
    BaseCrossValidator,
    KFold,
    StratifiedKFold,
)

from evaluation.tasks.base import Kind
from evaluation.tasks.metrics import classification_metrics, regression_metrics

MetricFn = Callable[..., float | np.ndarray]


@dataclass
class ColumnTask:
    """Predict a single column of an HF dataset from frozen image features."""

    name: str
    kind: Kind
    data: HFDataset
    splitter: BaseCrossValidator | list[tuple[np.ndarray, np.ndarray]] | None = None
    metric_fns: Sequence[MetricFn] | None = None
    image_column: str = "image"
    target_column: str = "target"
    group_column: str | None = None
    n_splits: int = 5
    seed: int = 0
    positive_label: object | None = None

    def __post_init__(self):
        targets = np.asarray(self.data[self.target_column])
        present = pd.notna(targets)
        valid = present if present.ndim == 1 else present.all(axis=tuple(range(1, present.ndim)))
        if not valid.all():
            self.data = self.data.select(np.flatnonzero(valid))

    def dataset(self) -> HFDataset:
        column_mapping = {self.image_column: "image", self.target_column: "target"}
        dataset = self.data.select_columns(list(column_mapping)).rename_columns(column_mapping)
        return dataset

    def split(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if isinstance(self.splitter, list):
            yield from self.splitter
            return

        splitter = self.splitter or self._default_splitter()
        indices = np.arange(len(self.data))
        targets = np.asarray(self.data[self.target_column])
        groups = np.asarray(self.data[self.group_column]) if self.group_column else None
        yield from splitter.split(indices, y=targets, groups=groups)

    def _default_splitter(self) -> BaseCrossValidator:
        kwargs = {"n_splits": self.n_splits, "shuffle": True, "random_state": self.seed}
        if self.kind == "regression":
            return KFold(**kwargs)
        return StratifiedKFold(**kwargs)

    def metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        test_idx: np.ndarray,
        y_score: np.ndarray | None = None,
    ) -> dict:
        if self.metric_fns is not None:
            scores = {}
            for metric_fn in self.metric_fns:
                value = metric_fn(
                    y_true,
                    y_pred,
                    test_idx=test_idx,
                    y_score=y_score,
                    positive_label=self.positive_label,
                )
                scores[metric_fn.__name__] = float(value)
            return scores
        if self.kind == "regression":
            return regression_metrics(y_true, y_pred)
        return classification_metrics(y_true, y_pred)
