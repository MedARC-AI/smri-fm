import importlib
import logging
import pkgutil
from inspect import signature
from typing import Callable

import evaluation.tasks
from evaluation.tasks.base import Task

_logger = logging.getLogger(__package__)

_TASK_REGISTRY: dict[str, Callable[..., Task]] = {}


def register_task(name_or_func: str | Callable[..., Task] | None = None):
    """Register a task builder.

    ```
    @register_task
    def new_task(**kwargs) -> Task:
        ...
    ```
    """

    def _decorator(func: Callable):
        name = name_or_func if isinstance(name_or_func, str) else func.__name__
        if name in _TASK_REGISTRY:
            _logger.warning(f"Task {name} already registered; overwriting.")
        _TASK_REGISTRY[name] = func
        return func

    if isinstance(name_or_func, Callable):
        return _decorator(name_or_func)
    return _decorator


def create_task(name: str, *, default_seed: int | None = None, **kwargs) -> Task:
    if name not in _TASK_REGISTRY:
        raise ValueError(f"Task {name} not registered")
    task_builder = _TASK_REGISTRY[name]
    if default_seed is not None and "seed" in signature(task_builder).parameters:
        kwargs.setdefault("seed", default_seed)
    return task_builder(**kwargs)


def list_tasks() -> list[str]:
    return list(_TASK_REGISTRY)


def import_task_plugins():
    """Finds and imports all modules registering new tasks."""
    plugins = {}
    for finder, name, ispkg in pkgutil.iter_modules(evaluation.tasks.__path__):
        if not (name in {"base", "registry", "resources"} or name.startswith("test_")):
            try:
                plugins[name] = importlib.import_module(f"evaluation.tasks.{name}")
            except Exception as exc:
                _logger.warning(f"Import task plugin {name} failed: {exc}")
    return plugins


# import all discovered plugins to register
_TASK_PLUGINS = import_task_plugins()
