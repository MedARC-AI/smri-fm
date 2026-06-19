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
DEFAULT_GENERATOR_BACKEND = "nv_generate_ctmr"
DEFAULT_WAVEDIT_CHECKPOINT_REPO = "danesed/WaveDiT"
DEFAULT_WAVEDIT_CHECKPOINT_FILENAME = "WaveDiT-Base.pth"
DEFAULT_WAVEDIT_CHECKPOINT_REVISION = "main"
DEFAULT_WAVEDIT_AGES = (6.0, 18.0, 30.0, 45.0, 60.0, 75.0, 90.0)
DEFAULT_WAVEDIT_OUTPUT_SIZE = (182, 218, 182)

ALLOWED_GENERATOR_BACKENDS = {"nv_generate_ctmr", "wavedit"}
ALLOWED_QC_MODES = {"direct_synthseg", "preprocess_then_synthseg"}
ALLOWED_QC_METRICS = {"min", "mean"}
ALLOWED_WAVEDIT_SAMPLERS = {"heun", "euler"}


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
class WaveDiTConfig:
    ages: tuple[float, ...] = DEFAULT_WAVEDIT_AGES
    checkpoint_path: Path | None = None
    checkpoint_repo: str = DEFAULT_WAVEDIT_CHECKPOINT_REPO
    checkpoint_filename: str = DEFAULT_WAVEDIT_CHECKPOINT_FILENAME
    checkpoint_revision: str = DEFAULT_WAVEDIT_CHECKPOINT_REVISION
    num_flow_steps: int = 10
    sampler: str = "heun"
    cfg_scale: float = 1.0
    cfg_rescale: float = 0.7
    morpheus_scale: float | None = None
    device: str = "cuda"


@dataclass(frozen=True)
class PipelineConfig:
    generator_backend: str
    generator_repo: Path
    output_dir: Path
    num_images: int
    random_seed: int
    output_size: tuple[int, int, int] | None
    targets: TargetSelection
    qc: QCConfig
    push_to_hf: PushConfig
    generator_python: tuple[str, ...]
    wavedit: WaveDiTConfig
    num_gpus: int = 1


def _as_string_tuple(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list.")
    if not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"{name} must contain only non-empty strings.")
    return tuple(value)


def _as_float_tuple(value: Any, name: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list.")
    floats: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise ValueError(f"{name} must contain only numbers.")
        floats.append(float(item))
    return tuple(floats)


def _as_size(value: Any, name: str) -> tuple[int, int, int] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{name} must be a list of three positive integers or null.")
    size: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(f"{name} must be a list of three positive integers or null.")
        if item < 1:
            raise ValueError(f"{name} values must be >= 1.")
        size.append(item)
    return (size[0], size[1], size[2])


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


def _as_optional_path(value: Any, name: str) -> Path | None:
    if value is None:
        return None
    return _as_path(value, name)


def _as_nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string.")
    return value


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

    raw_wavedit = raw.get("wavedit") or {}
    if not isinstance(raw_wavedit, dict):
        raise ValueError("wavedit must be a mapping.")

    generator_backend_value = raw.get("generator_backend", DEFAULT_GENERATOR_BACKEND)
    generator_backend_selected = _as_nonempty_string(generator_backend_value, "generator_backend")
    if generator_backend_selected not in ALLOWED_GENERATOR_BACKENDS:
        raise ValueError(f"generator_backend must be one of {sorted(ALLOWED_GENERATOR_BACKENDS)}.")

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

    selected_wavedit_sampler = raw_wavedit.get("sampler", "heun")
    if selected_wavedit_sampler not in ALLOWED_WAVEDIT_SAMPLERS:
        raise ValueError(f"wavedit.sampler must be one of {sorted(ALLOWED_WAVEDIT_SAMPLERS)}.")

    cfg = PipelineConfig(
        generator_backend=generator_backend_selected,
        generator_repo=(
            generator_repo or _as_path(raw.get("generator_repo"), "generator_repo")
        ).resolve(),
        output_dir=(output_dir or _as_path(raw.get("output_dir"), "output_dir")).resolve(),
        num_images=(
            num_images
            if num_images is not None
            else _as_int(raw.get("num_images"), "num_images", minimum=1)
        ),
        random_seed=_as_int(raw.get("random_seed", 1234), "random_seed"),
        output_size=_as_size(raw.get("output_size"), "output_size"),
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
        wavedit=WaveDiTConfig(
            ages=_as_float_tuple(
                raw_wavedit.get("ages", list(DEFAULT_WAVEDIT_AGES)),
                "wavedit.ages",
            ),
            checkpoint_path=(
                _as_optional_path(raw_wavedit.get("checkpoint_path"), "wavedit.checkpoint_path")
            ),
            checkpoint_repo=_as_nonempty_string(
                raw_wavedit.get("checkpoint_repo", DEFAULT_WAVEDIT_CHECKPOINT_REPO),
                "wavedit.checkpoint_repo",
            ),
            checkpoint_filename=_as_nonempty_string(
                raw_wavedit.get("checkpoint_filename", DEFAULT_WAVEDIT_CHECKPOINT_FILENAME),
                "wavedit.checkpoint_filename",
            ),
            checkpoint_revision=_as_nonempty_string(
                raw_wavedit.get("checkpoint_revision", DEFAULT_WAVEDIT_CHECKPOINT_REVISION),
                "wavedit.checkpoint_revision",
            ),
            num_flow_steps=_as_int(
                raw_wavedit.get("num_flow_steps", 10),
                "wavedit.num_flow_steps",
                minimum=1,
            ),
            sampler=selected_wavedit_sampler,
            cfg_scale=_as_optional_float(raw_wavedit.get("cfg_scale", 1.0), "wavedit.cfg_scale"),
            cfg_rescale=_as_optional_float(
                raw_wavedit.get("cfg_rescale", 0.7),
                "wavedit.cfg_rescale",
            ),
            morpheus_scale=_as_optional_float(
                raw_wavedit.get("morpheus_scale"),
                "wavedit.morpheus_scale",
            ),
            device=_as_nonempty_string(raw_wavedit.get("device", "cuda"), "wavedit.device"),
        ),
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
    if len(set(cfg.wavedit.ages)) != len(cfg.wavedit.ages):
        raise ValueError("wavedit.ages must not contain duplicate values.")
    if cfg.wavedit.cfg_scale is None:
        raise ValueError("wavedit.cfg_scale must be a number.")
    if cfg.wavedit.cfg_rescale is None:
        raise ValueError("wavedit.cfg_rescale must be a number.")
