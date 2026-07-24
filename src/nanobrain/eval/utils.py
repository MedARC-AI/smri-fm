"""Small cross-cutting helpers: name registries, seeding, git SHA, logging setup."""

import logging
import random
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger("nanobrain.eval")


def make_registry() -> tuple[Callable, Callable, Callable]:
    """A name -> builder registry, shared by models and tasks."""
    registry: dict[str, Callable] = {}

    def register(builder: Callable) -> Callable:
        assert builder.__name__ not in registry, f"duplicate registration: {builder.__name__!r}"
        registry[builder.__name__] = builder
        return builder

    def create(name: str, **kwargs):
        if name not in registry:
            raise KeyError(f"unknown name {name!r}; registered: {sorted(registry)}")
        return registry[name](**kwargs)

    def names() -> list[str]:
        return sorted(registry)

    return register, create, names


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
