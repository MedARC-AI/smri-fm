from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generic, TypeVar

from torch import nn

from evaluation.config_schema import ProbeConfig, TaskConfig

TaskConfigT = TypeVar("TaskConfigT", bound=TaskConfig)


class EvalTask(ABC, Generic[TaskConfigT]):
    id: str
    name: str
    output_name: str
    config_type: type[TaskConfigT] = TaskConfig

    @abstractmethod
    def run_probe(
        self,
        backbone: nn.Module,
        output_dir: Path,
        data_dir: Path,
        seed: int,
        device: str,
        task_config: TaskConfigT,
        probe_config: ProbeConfig,
    ) -> dict[str, Any]:
        """Run this task in frozen-backbone probe mode."""

    def parse_config(self, config: dict[str, Any] | TaskConfigT | None) -> TaskConfigT:
        return self.config_type.from_mapping(config)
