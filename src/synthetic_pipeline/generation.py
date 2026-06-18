from __future__ import annotations

import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Any

from .config import PipelineConfig

log = logging.getLogger("synthetic_pipeline.generation")

GENERATE_VERSION = "rflow-mr-brain"
TARGET_CONFIG = Path("configs/config_generate_mr_brain_default_fov_256_128.json")
ENV_TEMPLATE = Path("configs/environment_maisi_diff_model_rflow-mr-brain.json")
MODEL_DEF = Path("configs/config_network_rflow.json")

REQUIRED_GENERATOR_FILES = [
    TARGET_CONFIG,
    ENV_TEMPLATE,
    MODEL_DEF,
    Path("scripts/download_model_data.py"),
    Path("scripts/diff_model_infer_MANY.py"),
]


def generate(cfg: PipelineConfig) -> list[dict[str, Any]]:
    target_config, selected = validate_generation_inputs(cfg)
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
        write_runtime_model_config(target_config, target, target_seed, runtime_model)
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
            records.append(record_for_output(image_path, target, target_seed))

    return records


def validate_generation_inputs(
    cfg: PipelineConfig,
) -> tuple[dict[str, Any], list[tuple[int, dict[str, Any]]]]:
    validate_generator_repo(cfg.generator_repo)
    target_config = load_target_config(cfg.generator_repo / TARGET_CONFIG)
    selected = select_targets(target_config, cfg)
    if not selected:
        raise ValueError("No generator targets matched the requested conditions/modalities/planes.")
    return target_config, selected


def validate_generator_repo(repo: Path) -> None:
    if not repo.exists():
        raise FileNotFoundError(f"Generator repo not found: {repo}")
    missing = [str(path) for path in REQUIRED_GENERATOR_FILES if not (repo / path).exists()]
    if missing:
        raise FileNotFoundError(
            "Generator repo is missing required file(s): " + ", ".join(missing)
        )


def load_target_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        config = json.load(f)
    targets = config.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ValueError(f"{path} must define a non-empty targets list.")
    return config


def select_targets(target_config: dict[str, Any], cfg: PipelineConfig) -> list[tuple[int, dict[str, Any]]]:
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
