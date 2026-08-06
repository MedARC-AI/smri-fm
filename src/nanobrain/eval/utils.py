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
        if builder.__name__ in registry:
            logger.warning("overwriting the registered builder for %r", builder.__name__)
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
    kwargs = dict(cwd=Path(__file__).parent, capture_output=True, text=True, check=True)
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], **kwargs).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "-uno"], **kwargs).stdout.strip()
    return f"{sha}-dirty" if dirty else sha


def setup_logging(run_dir: Path) -> None:
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(run_dir / "log.txt")]
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    for handler in handlers:
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(handler)
    logger.propagate = False
