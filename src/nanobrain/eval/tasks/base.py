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
    # One name per foreground class, ordered by integer label (label 1 is class_names[0], ...);
    # 0 is always background. Drives the multiclass probe's per-class metrics.
    class_names: tuple[str, ...] = ("foreground",)


Task = RegressionTask | ClassificationTask | SegmentationTask
