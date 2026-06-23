from collections.abc import Iterator
from typing import Literal, Protocol

import numpy as np
from torch.utils.data import Dataset

Kind = Literal["regression", "classification"]


class Task(Protocol):
    name: str
    kind: Kind

    def dataset(self) -> Dataset:
        """Dataset of canonical ``{image, target}`` samples."""
        ...

    def split(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(train_idx, test_idx)`` splits; one pair, or many for outer CV."""
        ...

    def metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        test_idx: np.ndarray,
        y_score: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Score one fold; ``test_idx`` lets a task pull auxiliary metadata for scoring."""
        ...
