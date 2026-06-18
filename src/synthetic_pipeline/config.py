from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_SYNTHSEG_CMD = (
    "uvx --python 3.11 --from 'git+https://github.com/MedARC-AI/SynthSeg.git' SynthSeg"
)
DEFAULT_GENERATOR_PYTHON = "uv run --frozen python"

ALLOWED_QC_MODES = {"direct_synthseg", "preprocess_then_synthseg"}
ALLOWED_QC_METRICS = {"min", "mean"}


@dataclass(frozen=True)
class TargetSelection:
    conditions: tuple[str, ...]
    modalities: tuple[str, ...]
    planes: tuple[str, ...]


@dataclass(frozen=True)
class QCConfig:
    mode: str = "direct_synthseg"
    threshold: float | None = None
    metric: str = "min"
    synthseg_cmd: str = DEFAULT_SYNTHSEG_CMD
    threads: int = 8
    cpu: bool = False


@dataclass(frozen=True)
class PushConfig:
    enabled: bool = True


@dataclass(frozen=True)
class PipelineConfig:
    generator_repo: Path
    output_dir: Path
    num_images: int
    random_seed: int
    targets: TargetSelection
    qc: QCConfig
    push_to_hf: PushConfig
    generator_python: tuple[str, ...]
    num_gpus: int = 1


def _as_string_tuple(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list.")
    if not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"{name} must contain only non-empty strings.")
    return tuple(value)


def _as_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false.")
    return value


def _as_int(value: Any, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer.")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    return value


def _as_optional_float(value: Any, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a number or null.")
    return float(value)


def _as_path(value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty path string.")
    return Path(value).expanduser()


def load_config(
    path: Path,
    *,
    generator_repo: Path | None = None,
    output_dir: Path | None = None,
    num_images: int | None = None,
    qc_threshold: float | None = None,
    qc_metric: str | None = None,
    qc_mode: str | None = None,
) -> PipelineConfig:
    with path.open() as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError("Config file must contain a YAML mapping.")

    raw_targets = raw.get("targets") or {}
    if not isinstance(raw_targets, dict):
        raise ValueError("targets must be a mapping.")

    raw_qc = raw.get("qc") or {}
    if not isinstance(raw_qc, dict):
        raise ValueError("qc must be a mapping.")

    raw_push = raw.get("push_to_hf") or {}
    if not isinstance(raw_push, dict):
        raise ValueError("push_to_hf must be a mapping.")

    selected_qc_mode = qc_mode or raw_qc.get("mode", "direct_synthseg")
    if selected_qc_mode not in ALLOWED_QC_MODES:
        raise ValueError(f"qc.mode must be one of {sorted(ALLOWED_QC_MODES)}.")

    selected_qc_metric = qc_metric or raw_qc.get("metric", "min")
    if selected_qc_metric not in ALLOWED_QC_METRICS:
        raise ValueError(f"qc.metric must be one of {sorted(ALLOWED_QC_METRICS)}.")

    threshold = qc_threshold if qc_threshold is not None else raw_qc.get("threshold")
    synthseg_cmd = raw_qc.get("synthseg_cmd", DEFAULT_SYNTHSEG_CMD)
    if not isinstance(synthseg_cmd, str) or not synthseg_cmd:
        raise ValueError("qc.synthseg_cmd must be a non-empty string.")

    generator_python_text = raw.get("generator_python", DEFAULT_GENERATOR_PYTHON)
    if not isinstance(generator_python_text, str) or not generator_python_text:
        raise ValueError("generator_python must be a non-empty command string.")

    cfg = PipelineConfig(
        generator_repo=(generator_repo or _as_path(raw.get("generator_repo"), "generator_repo")).resolve(),
        output_dir=(output_dir or _as_path(raw.get("output_dir"), "output_dir")).resolve(),
        num_images=(
            num_images
            if num_images is not None
            else _as_int(raw.get("num_images"), "num_images", minimum=1)
        ),
        random_seed=_as_int(raw.get("random_seed", 1234), "random_seed"),
        targets=TargetSelection(
            conditions=_as_string_tuple(raw_targets.get("conditions"), "targets.conditions"),
            modalities=_as_string_tuple(raw_targets.get("modalities"), "targets.modalities"),
            planes=_as_string_tuple(raw_targets.get("planes"), "targets.planes"),
        ),
        qc=QCConfig(
            mode=selected_qc_mode,
            threshold=_as_optional_float(threshold, "qc.threshold"),
            metric=selected_qc_metric,
            synthseg_cmd=synthseg_cmd,
            threads=_as_int(raw_qc.get("threads", 8), "qc.threads", minimum=1),
            cpu=_as_bool(raw_qc.get("cpu", False), "qc.cpu"),
        ),
        push_to_hf=PushConfig(
            enabled=_as_bool(raw_push.get("enabled", True), "push_to_hf.enabled"),
        ),
        generator_python=tuple(shlex.split(generator_python_text)),
        num_gpus=_as_int(raw.get("num_gpus", 1), "num_gpus", minimum=1),
    )
    validate_config(cfg)
    return cfg


def validate_config(cfg: PipelineConfig) -> None:
    if cfg.num_images < 1:
        raise ValueError("num_images must be >= 1.")
    if cfg.num_gpus != 1:
        raise ValueError("Only num_gpus: 1 is supported in this pipeline version.")
    if cfg.qc.threshold is not None and cfg.qc.threshold < 0:
        raise ValueError("qc.threshold must be >= 0 when set.")
    if not cfg.generator_python:
        raise ValueError("generator_python parsed to an empty command.")

