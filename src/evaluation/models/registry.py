from __future__ import annotations

from collections.abc import Callable

from torch import nn

from evaluation.models.dummy import DummyBackbone

ModelFactory = Callable[..., nn.Module]

_MODEL_REGISTRY: dict[str, ModelFactory] = {
    "dummy": DummyBackbone,
}


def create_model(model: str | nn.Module, **kwargs) -> nn.Module:
    if isinstance(model, nn.Module):
        if kwargs:
            raise ValueError("model_kwargs cannot be used when model is already an nn.Module.")
        return model
    try:
        return _MODEL_REGISTRY[model](**kwargs)
    except KeyError as exc:
        available = ", ".join(list_models())
        raise ValueError(f"Unknown model '{model}'. Available models: {available}") from exc


def list_models() -> list[str]:
    return sorted(_MODEL_REGISTRY)
