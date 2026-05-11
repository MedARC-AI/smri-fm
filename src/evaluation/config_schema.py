from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from torch import nn


Profile = Literal["probe", "full"]


@dataclass(frozen=True)
class ProbeConfig:
    """Configuration for heads fit on top of frozen backbone embeddings."""

    head: str = "ridge"
    alpha: float = 1.0
    kwargs: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw_config: Mapping[str, Any] | "ProbeConfig" | None) -> "ProbeConfig":
        if raw_config is None:
            return cls()
        if isinstance(raw_config, cls):
            return raw_config

        config = dict(raw_config)
        kwargs = dict(config.pop("kwargs", {}))
        head = str(config.pop("head", cls.head))
        alpha = float(config.pop("alpha", cls.alpha))
        kwargs.update(config)
        return cls(head=head, alpha=alpha, kwargs=kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {"head": self.head, "alpha": self.alpha, "kwargs": dict(self.kwargs)}


@dataclass(frozen=True)
class TaskConfig:
    """Base task config.

    Concrete tasks should define a subclass when they need task-specific fields.
    The base class remains useful for future tasks whose config is not known yet.
    """

    source: str = "local"
    kwargs: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw_config: Mapping[str, Any] | "TaskConfig" | None) -> "TaskConfig":
        if raw_config is None:
            return cls()
        if isinstance(raw_config, cls):
            return raw_config

        config = dict(raw_config)
        kwargs = dict(config.pop("kwargs", {}))
        source = str(config.pop("source", cls.source))
        kwargs.update(config)
        return cls(source=source, kwargs=kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source, "kwargs": dict(self.kwargs)}


@dataclass(frozen=True)
class RunConfig:
    """Top-level evaluation run config consumed by the public API."""

    model: str | nn.Module = "dummy"
    profile: Profile = "probe"
    tasks: tuple[str, ...] = ("3",)
    output_dir: Path = Path("fomo26_runs")
    data_dir: Path = Path("fomo26_data")
    name: str | None = None
    seed: int = 4466
    device: str = "cpu"
    model_kwargs: Mapping[str, Any] = field(default_factory=dict)
    task_configs: Mapping[str, Mapping[str, Any] | TaskConfig] = field(default_factory=dict)
    probe_config: ProbeConfig = field(default_factory=ProbeConfig)

    @classmethod
    def from_mapping(cls, raw_config: Mapping[str, Any] | "RunConfig") -> "RunConfig":
        if isinstance(raw_config, cls):
            return raw_config

        config = dict(raw_config)
        legacy_task_kwargs = config.pop("task_kwargs", None)
        legacy_probe_kwargs = config.pop("probe_kwargs", None)

        task_configs = config.pop("task_configs", legacy_task_kwargs) or {}
        probe_config = config.pop("probe_config", legacy_probe_kwargs)

        return cls(
            model=config.pop("model", cls.model),
            profile=config.pop("profile", cls.profile),
            tasks=_normalize_task_ids(config.pop("tasks", cls.tasks)),
            output_dir=Path(config.pop("output_dir", cls.output_dir)),
            data_dir=Path(config.pop("data_dir", cls.data_dir)),
            name=config.pop("name", cls.name),
            seed=int(config.pop("seed", cls.seed)),
            device=str(config.pop("device", cls.device)),
            model_kwargs=dict(config.pop("model_kwargs", {})),
            task_configs=dict(task_configs),
            probe_config=ProbeConfig.from_mapping(probe_config),
        )

    def to_dict(self) -> dict[str, Any]:
        model = self.model if isinstance(self.model, str) else self.model.__class__.__name__
        return {
            "model": model,
            "profile": self.profile,
            "tasks": list(self.tasks),
            "output_dir": str(self.output_dir),
            "data_dir": str(self.data_dir),
            "name": self.name,
            "seed": self.seed,
            "device": self.device,
            "model_kwargs": dict(self.model_kwargs),
            "task_configs": {
                task_id: _to_config_dict(config)
                for task_id, config in self.task_configs.items()
            },
            "probe_config": self.probe_config.to_dict(),
        }


def _normalize_task_ids(tasks: Sequence[str | int] | str | int) -> tuple[str, ...]:
    if isinstance(tasks, (str, int)):
        return (str(tasks),)
    return tuple(str(task) for task in tasks)


def _to_config_dict(config: Mapping[str, Any] | TaskConfig) -> dict[str, Any]:
    if isinstance(config, TaskConfig):
        return config.to_dict()
    return dict(config)
