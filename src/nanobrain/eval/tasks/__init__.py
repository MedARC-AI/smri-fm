from nanobrain.eval.utils import make_registry

register_task, create_task, list_tasks = make_registry()

from nanobrain.eval.tasks import (  # noqa: E402,F401  (register on import)
    abide,
    adhd200,
    adni,
    cnp,
    dlbs,
    fomo,
)
