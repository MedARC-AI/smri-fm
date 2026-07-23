"""Task specs: a lazily-loaded dataset plus the columns each probe reads."""

from collections.abc import Callable
from dataclasses import dataclass

from datasets import Dataset

DatasetFn = Callable[[], Dataset]


@dataclass
class RegressionTask:
    name: str
    dataset_fn: DatasetFn
    target_col: str
    image_col: str = "image"


@dataclass
class ClassificationTask:
    name: str
    dataset_fn: DatasetFn
    target_col: str
    image_col: str = "image"
    target_map: dict | None = None


@dataclass
class SegmentationTask:
    name: str
    dataset_fn: DatasetFn
    seg_col: str
    image_col: str = "image"


Task = RegressionTask | ClassificationTask | SegmentationTask
