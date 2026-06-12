from __future__ import annotations

import importlib
import logging
import pkgutil
from collections.abc import Callable

import eval.datasets

_logger = logging.getLogger(__package__)
_DATASET_REGISTRY: dict[str, Callable] = {}


def register_dataset(name_or_func=None):
    def decorator(func):
        name = name_or_func if isinstance(name_or_func, str) else func.__name__
        if name in _DATASET_REGISTRY:
            _logger.warning("Dataset %s already registered; overwriting", name)
        _DATASET_REGISTRY[name] = func
        return func
    if callable(name_or_func):
        return decorator(name_or_func)
    return decorator


def create_dataset(name: str, **kwargs):
    if name not in _DATASET_REGISTRY:
        raise ValueError(f"Dataset {name!r} not registered; available: {sorted(_DATASET_REGISTRY)}")
    return _DATASET_REGISTRY[name](**kwargs)


def list_datasets() -> list[str]:
    return sorted(_DATASET_REGISTRY)


def import_dataset_plugins():
    plugins = {}
    for _, name, _ in pkgutil.iter_modules(eval.datasets.__path__):
        if name in {"base", "registry", "template"} or name.startswith("test_"):
            continue
        try:
            plugins[name] = importlib.import_module(f"eval.datasets.{name}")
        except Exception as exc:
            _logger.warning("Import dataset plugin %s failed: %s", name, exc)
    return plugins


_DATASET_PLUGINS = import_dataset_plugins()
