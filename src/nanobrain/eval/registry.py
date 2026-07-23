"""Tiny name -> builder registries, shared by models and tasks."""

from collections.abc import Callable


def make_registry() -> tuple[Callable, Callable, Callable]:
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
