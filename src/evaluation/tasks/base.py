from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from torch import nn


class EvalTask(ABC):
    id: str
    name: str
    output_name: str

    @abstractmethod
    def run_probe(
        self,
        backbone: nn.Module,
        output_dir: Path,
        data_dir: Path,
        seed: int,
        device: str,
        task_kwargs: dict[str, Any],
        probe_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Run this task in frozen-backbone probe mode."""
