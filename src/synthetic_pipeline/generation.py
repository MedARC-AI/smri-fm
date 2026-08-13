from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from .config import DEFAULT_WAVEDIT_OUTPUT_SIZE, PipelineConfig

log = logging.getLogger("synthetic_pipeline.generation")

BACKEND_NV_GENERATE_CTMR = "nv_generate_ctmr"
BACKEND_WAVEDIT = "wavedit"

GENERATE_VERSION = "rflow-mr-brain"
TARGET_CONFIG = Path("configs/config_generate_mr_brain_default_fov_256_128.json")
ENV_TEMPLATE = Path("configs/environment_maisi_diff_model_rflow-mr-brain.json")
MODEL_DEF = Path("configs/config_network_rflow.json")
NV_OUTPUT_SIZE_DIVISOR = 4

REQUIRED_NV_GENERATOR_FILES = [
    TARGET_CONFIG,
    ENV_TEMPLATE,
    MODEL_DEF,
    Path("scripts/download_model_data.py"),
    Path("scripts/diff_model_infer_MANY.py"),
]
REQUIRED_WAVEDIT_GENERATOR_FILES = [
    Path("scripts/generate.py"),
    Path("wavedit"),
]

WAVEDIT_TARGET = {
    "condition": "whole_brain",
    "modality_name": "mri_t1",
    "modality": 9,
    "plane": "axial",
    "spacing": [1, 1, 1],
    "output_subdir": "whole_brain/mri_t1/axial",
}


def generate(cfg: PipelineConfig) -> list[dict[str, Any]]:
    if cfg.generator_backend == BACKEND_NV_GENERATE_CTMR:
        return generate_nv_generate_ctmr(cfg)
    if cfg.generator_backend == BACKEND_WAVEDIT:
        return generate_wavedit(cfg)
    raise ValueError(f"Unsupported generator backend: {cfg.generator_backend}")


def generate_nv_generate_ctmr(cfg: PipelineConfig) -> list[dict[str, Any]]:
    target_config, selected = validate_nv_generation_inputs(cfg)
    runtime_dir = cfg.output_dir / "runtime_configs"
    generated_root = cfg.output_dir / "generated"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    generated_root.mkdir(parents=True, exist_ok=True)

    download_models(cfg)

    records: list[dict[str, Any]] = []
    for original_index, target in selected:
        target_seed = cfg.random_seed + original_index * 100000
        output_subdir = target["output_subdir"]
        output_dir = generated_root / output_subdir
        output_dir.mkdir(parents=True, exist_ok=True)

        runtime_model = runtime_dir / f"target_{original_index:03d}_model.json"
        runtime_env = runtime_dir / f"target_{original_index:03d}_env.json"
        runtime_target = resolve_nv_runtime_target(cfg, target)
        write_runtime_model_config(target_config, runtime_target, target_seed, runtime_model)
        write_runtime_env_config(cfg.generator_repo / ENV_TEMPLATE, output_dir, runtime_env)

        before = set(output_dir.glob("*.nii.gz"))
        run_generator_target(cfg, runtime_env, runtime_model)
        after = set(output_dir.glob("*.nii.gz"))
        new_paths = sorted(after - before)
        if len(new_paths) != cfg.num_images:
            raise RuntimeError(
                f"Expected {cfg.num_images} new image(s) for target {output_subdir}, "
                f"found {len(new_paths)}."
            )

        for image_path in new_paths:
            records.append(record_for_output(image_path, runtime_target, target_seed))

    return records


def generate_wavedit(cfg: PipelineConfig) -> list[dict[str, Any]]:
    validate_wavedit_generation_inputs(cfg)
    checkpoint = resolve_wavedit_checkpoint(cfg)
    generated_root = cfg.output_dir / "generated"
    output_size = cfg.output_size or DEFAULT_WAVEDIT_OUTPUT_SIZE
    records: list[dict[str, Any]] = []

    for age_index, age, per_age_count in distribute_wavedit_images(cfg):
        if per_age_count < 1:
            continue
        target_seed = cfg.random_seed + age_index * 100000
        target = wavedit_target_for_age(age, output_size)
        output_dir = generated_root / WAVEDIT_TARGET["output_subdir"]
        condition_dir = generated_root / target["output_subdir"]
        output_dir.mkdir(parents=True, exist_ok=True)

        before = set(condition_dir.rglob("*.nii.gz")) if condition_dir.exists() else set()
        run_wavedit_target(
            cfg,
            checkpoint,
            output_dir,
            age,
            per_age_count,
            target_seed,
            output_size,
        )
        after = set(condition_dir.rglob("*.nii.gz")) if condition_dir.exists() else set()
        new_paths = sorted(after - before)
        if len(new_paths) != per_age_count:
            raise RuntimeError(
                f"Expected {per_age_count} new WaveDiT image(s) for age {age:g}, "
                f"found {len(new_paths)}."
            )

        for image_path in new_paths:
            records.append(record_for_output(image_path, target, target_seed))

    if len(records) != cfg.num_images:
        raise RuntimeError(f"Expected {cfg.num_images} WaveDiT image(s), found {len(records)}.")
    return records


def validate_generation_inputs(
    cfg: PipelineConfig,
) -> Any:
    if cfg.generator_backend == BACKEND_NV_GENERATE_CTMR:
        return validate_nv_generation_inputs(cfg)
    if cfg.generator_backend == BACKEND_WAVEDIT:
        return validate_wavedit_generation_inputs(cfg)
    raise ValueError(f"Unsupported generator backend: {cfg.generator_backend}")


def validate_nv_generation_inputs(
    cfg: PipelineConfig,
) -> tuple[dict[str, Any], list[tuple[int, dict[str, Any]]]]:
    validate_nv_generator_repo(cfg.generator_repo)
    target_config = load_target_config(cfg.generator_repo / TARGET_CONFIG)
    selected = select_targets(target_config, cfg)
    if not selected:
        raise ValueError("No generator targets matched the requested conditions/modalities/planes.")
    if cfg.output_size is not None:
        validate_nv_output_size(cfg.output_size)
    return target_config, selected


def validate_nv_generator_repo(repo: Path) -> None:
    if not repo.exists():
        raise FileNotFoundError(f"Generator repo not found: {repo}")
    missing = [str(path) for path in REQUIRED_NV_GENERATOR_FILES if not (repo / path).exists()]
    if missing:
        raise FileNotFoundError(
            "NV generator repo is missing required file(s): " + ", ".join(missing)
        )


def validate_wavedit_generation_inputs(cfg: PipelineConfig) -> None:
    validate_wavedit_generator_repo(cfg.generator_repo)
    if cfg.targets.conditions != ("whole_brain",):
        raise ValueError("WaveDiT requires targets.conditions: [whole_brain].")
    if cfg.targets.modalities != ("mri_t1",):
        raise ValueError("WaveDiT requires targets.modalities: [mri_t1].")
    if cfg.targets.planes != ("axial",):
        raise ValueError("WaveDiT requires targets.planes: [axial].")


def validate_wavedit_generator_repo(repo: Path) -> None:
    if not repo.exists():
        raise FileNotFoundError(f"WaveDiT repo not found: {repo}")
    missing = [str(path) for path in REQUIRED_WAVEDIT_GENERATOR_FILES if not (repo / path).exists()]
    if missing:
        raise FileNotFoundError(
            "WaveDiT repo is missing required file(s): " + ", ".join(missing)
        )


def validate_nv_output_size(output_size: tuple[int, int, int]) -> None:
    if any(dim % NV_OUTPUT_SIZE_DIVISOR != 0 for dim in output_size):
        raise ValueError(
            "output_size values must be divisible by "
            f"{NV_OUTPUT_SIZE_DIVISOR} for the NV backend."
        )


def load_target_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        config = json.load(f)
    targets = config.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ValueError(f"{path} must define a non-empty targets list.")
    return config


def select_targets(
    target_config: dict[str, Any],
    cfg: PipelineConfig,
) -> list[tuple[int, dict[str, Any]]]:
    selected: list[tuple[int, dict[str, Any]]] = []
    for index, target in enumerate(target_config["targets"]):
        if target.get("condition") not in cfg.targets.conditions:
            continue
        if target.get("modality_name") not in cfg.targets.modalities:
            continue
        if target.get("plane") not in cfg.targets.planes:
            continue
        validate_target(target, index)
        selected.append((index, target))
    return selected


def validate_target(target: dict[str, Any], index: int) -> None:
    required = {
        "condition",
        "modality_name",
        "modality",
        "plane",
        "dim",
        "spacing",
        "fov",
        "output_subdir",
    }
    missing = sorted(required - set(target))
    if missing:
        raise ValueError(f"Generator target {index} is missing key(s): {missing}")


def resolve_nv_runtime_target(cfg: PipelineConfig, target: dict[str, Any]) -> dict[str, Any]:
    runtime_target = dict(target)
    if cfg.output_size is None:
        return runtime_target

    output_size = list(cfg.output_size)
    fov = target["fov"]
    runtime_target["dim"] = output_size
    runtime_target["spacing"] = [float(fov_value) / dim for fov_value, dim in zip(fov, output_size)]
    return runtime_target


def download_models(cfg: PipelineConfig) -> None:
    cmd = [
        *cfg.generator_python,
        "-m",
        "scripts.download_model_data",
        "--version",
        GENERATE_VERSION,
        "--root_dir",
        "./",
        "--model_only",
    ]
    log.info("Downloading/checking generator model weights.")
    subprocess.run(cmd, cwd=cfg.generator_repo, check=True)


def write_runtime_model_config(
    target_config: dict[str, Any],
    target: dict[str, Any],
    target_seed: int,
    output_path: Path,
) -> None:
    inference = dict(target_config.get("diffusion_unet_inference", {}))
    inference["dim"] = target["dim"]
    inference["spacing"] = target["spacing"]
    inference["modality"] = int(target["modality"])
    inference["random_seed"] = target_seed

    runtime_config = {
        "diffusion_unet_train": target_config.get("diffusion_unet_train", {}),
        "diffusion_unet_inference": inference,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(runtime_config, f, indent=4)
        f.write("\n")


def write_runtime_env_config(template_path: Path, output_dir: Path, output_path: Path) -> None:
    with template_path.open() as f:
        env_config = json.load(f)
    env_config["output_dir"] = str(output_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(env_config, f, indent=4)
        f.write("\n")


def run_generator_target(cfg: PipelineConfig, runtime_env: Path, runtime_model: Path) -> None:
    cmd = [
        *cfg.generator_python,
        "-m",
        "scripts.diff_model_infer_MANY",
        "-t",
        str((cfg.generator_repo / MODEL_DEF).resolve()),
        "-e",
        str(runtime_env),
        "-c",
        str(runtime_model),
        "-g",
        str(cfg.num_gpus),
        "--num_images",
        str(cfg.num_images),
    ]
    log.info("Generating %d image(s): %s", cfg.num_images, runtime_model.name)
    subprocess.run(cmd, cwd=cfg.generator_repo, check=True)


def resolve_wavedit_checkpoint(cfg: PipelineConfig) -> Path:
    if cfg.wavedit.checkpoint_path is not None:
        checkpoint_path = cfg.wavedit.checkpoint_path.resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"WaveDiT checkpoint not found: {checkpoint_path}")
        return checkpoint_path

    from huggingface_hub import hf_hub_download

    checkpoint_path = hf_hub_download(
        repo_id=cfg.wavedit.checkpoint_repo,
        filename=cfg.wavedit.checkpoint_filename,
        revision=cfg.wavedit.checkpoint_revision,
    )
    return Path(checkpoint_path)


def distribute_wavedit_images(cfg: PipelineConfig) -> list[tuple[int, float, int]]:
    base = cfg.num_images // len(cfg.wavedit.ages)
    remainder = cfg.num_images % len(cfg.wavedit.ages)
    return [
        (index, age, base + (1 if index < remainder else 0))
        for index, age in enumerate(cfg.wavedit.ages)
    ]


def wavedit_target_for_age(age: float, output_size: tuple[int, int, int]) -> dict[str, Any]:
    target = dict(WAVEDIT_TARGET)
    target["dim"] = list(output_size)
    target["fov"] = list(output_size)
    target["output_subdir"] = f"{WAVEDIT_TARGET['output_subdir']}/{wavedit_condition_tag(age)}"
    return target


def wavedit_condition_tag(age: float) -> str:
    return f"age_{age:.2f}"


def run_wavedit_target(
    cfg: PipelineConfig,
    checkpoint: Path,
    output_dir: Path,
    age: float,
    num_images: int,
    seed: int,
    output_size: tuple[int, int, int],
) -> None:
    cmd = [
        *cfg.generator_python,
        "scripts/generate.py",
        str(checkpoint),
        str(output_dir),
        "--cfg-scale",
        str(cfg.wavedit.cfg_scale),
        "--cfg-rescale",
        str(cfg.wavedit.cfg_rescale),
        "--num-flow-steps",
        str(cfg.wavedit.num_flow_steps),
        "--sampler",
        cfg.wavedit.sampler,
        "--save-size",
        *(str(value) for value in output_size),
        "--seed",
        str(seed),
        "--device",
        cfg.wavedit.device,
    ]
    if cfg.wavedit.morpheus_scale is not None:
        cmd.extend(["--morpheus-scale", str(cfg.wavedit.morpheus_scale)])
    cmd.extend(
        [
            "specific",
            "--conditions",
            f"age={age:g}",
            "--num-samples",
            str(num_images),
        ]
    )
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(cfg.generator_repo)
        if not existing_pythonpath
        else f"{cfg.generator_repo}{os.pathsep}{existing_pythonpath}"
    )
    log.info("Generating %d WaveDiT image(s) for age %s.", num_images, f"{age:g}")
    subprocess.run(cmd, cwd=cfg.generator_repo, check=True, env=env)


def record_for_output(image_path: Path, target: dict[str, Any], target_seed: int) -> dict[str, Any]:
    seed = _seed_from_filename(image_path.name)
    return {
        "image_path": str(image_path.resolve()),
        "condition": target["condition"],
        "modality": target["modality_name"],
        "modality_code": int(target["modality"]),
        "plane": target["plane"],
        "dim": "x".join(str(v) for v in target["dim"]),
        "spacing": "x".join(str(v) for v in target["spacing"]),
        "fov": "x".join(str(v) for v in target["fov"]),
        "target_output_subdir": target["output_subdir"],
        "generation_seed": seed if seed is not None else target_seed,
        "qc_path": None,
        "qc_min": None,
        "qc_mean": None,
        "qc_metric": None,
        "qc_threshold": None,
        "qc_pass": None,
        "qc_error": None,
    }


def _seed_from_filename(name: str) -> int | None:
    match = re.search(r"_seed(\d+)_", name)
    if match is None:
        return None
    return int(match.group(1))
