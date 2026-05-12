from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf
from torch import nn

from evaluation.config_schema import ProbeConfig, RunConfig, TaskConfig
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
    task_configs: Mapping[str, Mapping[str, Any] | TaskConfig] | None = None,
    probe_config: Mapping[str, Any] | ProbeConfig | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run one or more FOMO26-style local evaluations.

    The first implemented profile is ``probe``: the supplied model is treated as a
    frozen backbone, embeddings are extracted without gradients, and task-specific
    heads are fit on top.
    """

    run_config = RunConfig.from_mapping(
        {
            "model": model,
            "profile": profile,
            "tasks": tasks or DEFAULT_TASKS,
            "output_dir": output_dir,
            "data_dir": data_dir,
            "name": name,
            "seed": seed,
            "model_kwargs": model_kwargs or {},
            "task_configs": task_configs,
            "probe_config": probe_config,
            "device": device,
        }
    )

    if run_config.profile != "probe":
        raise NotImplementedError("Only profile='probe' is implemented.")

    task_ids = list(run_config.tasks)
    backbone = create_model(run_config.model, **dict(run_config.model_kwargs))

    # Freeze the backbone in probe mode, train in full mode
    if run_config.profile == "probe":
        _freeze_backbone(backbone)
    else:
        backbone.train()

    run_id = _make_unique_run_id(
        output_dir=run_config.output_dir,
        profile=run_config.profile,
        model=run_config.model,
        tasks=task_ids,
        label=run_config.name,
    )
    run_dir = run_config.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    config = run_config.to_dict()
    config["run_id"] = run_id
    _write_json(run_dir / "config.json", config)

    results: dict[str, Any] = {"run_id": run_id, "run_dir": str(run_dir), "tasks": {}}
    for task_id in task_ids:
        task = create_task(task_id)
        task_config = task.parse_config(run_config.task_configs.get(task_id))
        task_run_dir = run_dir / task.output_name
        task_result = task.run_probe(
            backbone=backbone,
            output_dir=task_run_dir,
            data_dir=run_config.data_dir,
            seed=run_config.seed,
            device=run_config.device,
            task_config=task_config,
            probe_config=run_config.probe_config,
        )
        results["tasks"][task_id] = task_result

    metrics = {
        task_id: task_result["metrics"]
        for task_id, task_result in results["tasks"].items()
    }
    _write_json(run_dir / "metrics.json", metrics)
    _write_json(
        run_dir / "run_metadata.json",
        _run_metadata(run_id=run_id, config=config, results=results),
    )
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


def _make_unique_run_id(
    output_dir: Path,
    profile: str,
    model: str | nn.Module,
    tasks: Sequence[str],
    label: str | None,
) -> str:
    base_run_id = _default_run_id(profile=profile, model=model, tasks=tasks, label=label)
    run_id = base_run_id
    suffix = 1
    while (output_dir / run_id).exists():
        run_id = f"{base_run_id}__{suffix:02d}"
        suffix += 1
    return run_id


def _default_run_id(
    profile: str,
    model: str | nn.Module,
    tasks: Sequence[str],
    label: str | None,
) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%SZ")
    model_name = model if isinstance(model, str) else model.__class__.__name__
    task_name = "-".join(_sanitize_path_part(task) for task in tasks)
    parts = [
        timestamp,
        _sanitize_path_part(label) if label else None,
        _sanitize_path_part(profile),
        f"tasks-{task_name}",
        _sanitize_path_part(model_name),
    ]
    return "__".join(part for part in parts if part)


def _sanitize_path_part(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-")
    return sanitized or "unnamed"


def _run_metadata(
    run_id: str,
    config: Mapping[str, Any],
    results: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "config": config,
        "task_outputs": {
            task_id: task_result["output_dir"]
            for task_id, task_result in results["tasks"].items()
        },
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
