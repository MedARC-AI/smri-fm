from __future__ import annotations

from evaluation.tasks.base import EvalTask
from evaluation.tasks.task3_brain_age import BrainAgeTask

_TASK_REGISTRY: dict[str, type[EvalTask]] = {
    "3": BrainAgeTask,
    "task3": BrainAgeTask,
    "brain_age": BrainAgeTask,
}


def create_task(task_id: str | int) -> EvalTask:
    key = str(task_id)
    try:
        return _TASK_REGISTRY[key]()
    except KeyError as exc:
        available = ", ".join(list_tasks())
        raise ValueError(f"Unknown task '{task_id}'. Available tasks: {available}") from exc


def list_tasks() -> list[str]:
    return sorted(_TASK_REGISTRY)
