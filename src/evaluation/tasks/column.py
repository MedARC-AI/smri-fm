from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd
from datasets import Dataset as HFDataset
from sklearn.model_selection import BaseCrossValidator, GroupKFold, StratifiedGroupKFold

from evaluation.tasks.base import Kind
from evaluation.tasks.metrics import classification_metrics, regression_metrics


@dataclass
class ColumnTask:
    """Predict a single column of an HF dataset from frozen image features."""

    name: str
    kind: Kind
    data: HFDataset
    group_column: str
    image_column: str = "image"
    target_column: str = "target"
    n_splits: int = 5
    seed: int = 0

    def __post_init__(self):
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

    def metrics(self, y_true: np.ndarray, y_pred: np.ndarray, test_idx: np.ndarray) -> dict:
        if self.kind == "regression":
            return regression_metrics(y_true, y_pred)
        return classification_metrics(y_true, y_pred)
