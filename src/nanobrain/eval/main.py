"""Entry point: resolve config, extract features once, run the probe for the task type."""

import argparse
import json
import logging
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from nanobrain.eval import probe as probe_module
from nanobrain.eval.models import create_model, list_models
from nanobrain.eval.tasks import create_task, list_tasks
from nanobrain.eval.tasks.base import ClassificationTask, RegressionTask, SegmentationTask

DEFAULT_CONFIG = Path(__file__).parent / "config.yaml"
logger = logging.getLogger("nanobrain.eval")


def run_probe(cfg, task, model, transform, device) -> dict:
    dataset = task.dataset_fn()
    logger.info(f"dataset: {len(dataset)} samples")
    cv = (cfg.n_splits, cfg.n_repeats, cfg.seed, cfg.n_boot)

    if isinstance(task, RegressionTask):
        X = probe_module.extract_global_features(
            model, transform, dataset, task.image_col, device, cfg.batch_size, cfg.num_workers
        )
        y = probe_module.read_targets(dataset, task.target_col)
        return probe_module.reg_probe(X, y, *cv)
    if isinstance(task, ClassificationTask):
        X = probe_module.extract_global_features(
            model, transform, dataset, task.image_col, device, cfg.batch_size, cfg.num_workers
        )
        y = probe_module.read_targets(dataset, task.target_col, task.target_map)
        return probe_module.cls_probe(X, y, *cv)
    if isinstance(task, SegmentationTask):
        features, fractions = probe_module.extract_patch_features(
            model, transform, dataset, task.image_col, task.seg_col, device
        )
        return probe_module.seg_probe(features, fractions, *cv)
    raise TypeError(f"unknown task type {type(task)}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def git_sha() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=Path(__file__).parent,
        capture_output=True,
        text=True,
    )
    return out.stdout.strip() or "unknown"


def setup_logging(run_dir: Path) -> None:
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(run_dir / "log.txt")]
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    for handler in handlers:
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(handler)
    logger.propagate = False


def main(
    model_name: str, task_name: str, config_path: str | None, overrides: list[str] | None
) -> dict:
    cfg = OmegaConf.load(DEFAULT_CONFIG)
    if config_path:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(config_path))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    OmegaConf.set_struct(cfg, True)

    cfg.name = cfg.name or f"{model_name}__{task_name}"
    run_dir = Path(cfg.output_root) / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir)
    set_seed(cfg.seed)

    sha = git_sha()
    logger.info(f"run {cfg.name} (git {sha})")
    logger.info(f"config:\n{OmegaConf.to_yaml(cfg).rstrip()}")
    OmegaConf.save(cfg, run_dir / "config.yaml")

    device = torch.device(cfg.device)
    task = create_task(task_name, **OmegaConf.to_container(cfg.task_kwargs, resolve=True))
    model, transform = create_model(
        model_name, **OmegaConf.to_container(cfg.model_kwargs, resolve=True)
    )
    model.to(device)
    model.eval()  # freeze BatchNorm/dropout so features are deterministic

    summary = run_probe(cfg, task, model, transform, device)
    record = {"model": model_name, "task": task_name, "git": sha, **summary}
    (run_dir / "metrics.jsonl").write_text(json.dumps(record) + "\n")
    logger.info("results: " + "  ".join(f"{k}={v:.4f}" for k, v in summary.items()))
    return record


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help=f"one of {list_models()}")
    parser.add_argument("task", help=f"one of {list_tasks()}")
    parser.add_argument("--config")
    parser.add_argument("--overrides", nargs="+")
    args = parser.parse_args()
    main(args.model, args.task, args.config, args.overrides)


if __name__ == "__main__":
    cli()
