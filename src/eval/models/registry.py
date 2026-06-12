import importlib
import logging
import pkgutil
from collections.abc import Callable

import eval.models
from eval.models.base import ModelTransformPair

_logger = logging.getLogger(__package__)
_MODEL_REGISTRY: dict[str, Callable[..., ModelTransformPair]] = {}


def register_model(name_or_func=None):
    def decorator(func):
        name = name_or_func if isinstance(name_or_func, str) else func.__name__
        if name in _MODEL_REGISTRY:
            _logger.warning("Model %s already registered; overwriting", name)
        _MODEL_REGISTRY[name] = func
        return func
    if callable(name_or_func):
        return decorator(name_or_func)
    return decorator


def create_model(name: str, **kwargs) -> ModelTransformPair:
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Model {name!r} not registered; available: {sorted(_MODEL_REGISTRY)}")
    result = _MODEL_REGISTRY[name](**kwargs)
    if isinstance(result, tuple):
        return result
    return None, result


def list_models() -> list[str]:
    return sorted(_MODEL_REGISTRY)


def import_model_plugins():
    plugins = {}
    for _, name, _ in pkgutil.iter_modules(eval.models.__path__):
        if name in {"base", "registry", "template"} or name.startswith("test_"):
            continue
        try:
            plugins[name] = importlib.import_module(f"eval.models.{name}")
        except Exception as exc:
            _logger.warning("Import model plugin %s failed: %s", name, exc)
    return plugins


_MODEL_PLUGINS = import_model_plugins()
