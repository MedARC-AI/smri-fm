from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd
from datasets import Dataset as HFDataset
from sklearn.model_selection import BaseCrossValidator

from evaluation.tasks.base import Kind
from evaluation.tasks.metrics import classification_metrics, regression_metrics


@dataclass
class ColumnTask:
    """Predict a single column of an HF dataset from frozen image features."""

    name: str
    kind: Kind
    data: HFDataset
    splitter: BaseCrossValidator | list[tuple[np.ndarray, np.ndarray]]
    image_column: str = "image"
    mask_column: str = "mask"
    target_column: str = "target"
    group_column: str | None = None

    def __post_init__(self):
        targets = np.asarray(self.data[self.target_column])
        valid = np.where(pd.notna(targets))[0]
        if len(valid) < len(self.data):
            self.data = self.data.select(valid)

    def dataset(self) -> HFDataset:
        column_mapping = {
            self.image_column: "image",
            self.mask_column: "mask",
            self.target_column: "target",
        }
        dataset = self.data.select_columns(list(column_mapping)).rename_columns(column_mapping)
        return dataset

    def split(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if isinstance(self.splitter, list):
            yield from self.splitter
            return

        indices = np.arange(len(self.data))
        targets = np.asarray(self.data[self.target_column])
        groups = np.asarray(self.data[self.group_column]) if self.group_column else None
        yield from self.splitter.split(indices, y=targets, groups=groups)

    def metrics(self, y_true: np.ndarray, y_pred: np.ndarray, test_idx: np.ndarray) -> dict:
        if self.kind == "regression":
            return regression_metrics(y_true, y_pred)
        return classification_metrics(y_true, y_pred)
