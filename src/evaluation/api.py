from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf
from torch import nn

from evaluation.models.registry import create_model
from evaluation.tasks.registry import create_task

DEFAULT_CONFIG = Path(__file__).parent / "config/default_probe.yaml"
DEFAULT_TASKS = ["3"]


def run_evals(
    model: str | nn.Module = "dummy",
    profile: str = "probe",
    tasks: Sequence[str | int] | None = None,
    output_dir: str | Path = "fomo26_runs",
    data_dir: str | Path = "fomo26_data",
    name: str | None = None,
    seed: int = 4466,
    model_kwargs: Mapping[str, Any] | None = None,
    task_kwargs: Mapping[str, Any] | None = None,
    probe_kwargs: Mapping[str, Any] | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run one or more FOMO26-style local evaluations.

    The first implemented profile is ``probe``: the supplied model is treated as a
    frozen backbone, embeddings are extracted without gradients, and task-specific
    heads are fit on top.
    """

    if profile != "probe":
        raise NotImplementedError("Only profile='probe' is implemented.")

    task_ids = [str(task) for task in (tasks or DEFAULT_TASKS)]
    backbone = create_model(model, **dict(model_kwargs or {}))
    _freeze_backbone(backbone)

    run_name = name or _default_run_name(profile=profile, model=model, tasks=task_ids)
    run_dir = Path(output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "model": model if isinstance(model, str) else model.__class__.__name__,
        "profile": profile,
        "tasks": task_ids,
        "output_dir": str(output_dir),
        "data_dir": str(data_dir),
        "name": run_name,
        "seed": seed,
        "model_kwargs": dict(model_kwargs or {}),
        "task_kwargs": dict(task_kwargs or {}),
        "probe_kwargs": dict(probe_kwargs or {}),
        "device": device,
    }
    _write_json(run_dir / "config.json", config)

    results: dict[str, Any] = {"run_dir": str(run_dir), "tasks": {}}
    for task_id in task_ids:
        task = create_task(task_id)
        task_run_dir = run_dir / task.output_name
        task_result = task.run_probe(
            backbone=backbone,
            output_dir=task_run_dir,
            data_dir=Path(data_dir),
            seed=seed,
            device=device,
            task_kwargs=dict(task_kwargs or {}).get(task_id, dict(task_kwargs or {})),
            probe_kwargs=dict(probe_kwargs or {}),
        )
        results["tasks"][task_id] = task_result

    metrics = {
        task_id: task_result["metrics"]
        for task_id, task_result in results["tasks"].items()
    }
    _write_json(run_dir / "metrics.json", metrics)
    _write_json(run_dir / "run_metadata.json", _run_metadata(config=config, results=results))
    return results


def run_evals_from_config(yaml_config: str | Path | Mapping[str, Any] | DictConfig) -> dict[str, Any]:
    """Load an evaluation config and pass it to :func:`run_evals`."""

    if isinstance(yaml_config, (str, Path)):
        cfg = OmegaConf.load(yaml_config)
    else:
        cfg = OmegaConf.create(yaml_config)

    base_cfg = OmegaConf.load(DEFAULT_CONFIG)
    cfg = OmegaConf.merge(base_cfg, cfg)
    params = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(params, dict):
        raise TypeError("Evaluation config must resolve to a mapping.")
    return run_evals(**params)


def _freeze_backbone(backbone: nn.Module) -> None:
    backbone.requires_grad_(False)
    backbone.eval()


def _default_run_name(profile: str, model: str | nn.Module, tasks: Sequence[str]) -> str:
    model_name = model if isinstance(model, str) else model.__class__.__name__
    task_name = "-".join(tasks)
    return f"eval_{profile}/{task_name}__{model_name}"


def _run_metadata(config: Mapping[str, Any], results: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "config": config,
        "task_outputs": {
            task_id: task_result["output_dir"]
            for task_id, task_result in results["tasks"].items()
        },
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
