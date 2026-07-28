from nanobrain.eval.utils import make_registry

register_task, create_task, list_tasks = make_registry()

from nanobrain.eval.tasks import adni, cnp, dlbs, fomo  # noqa: E402,F401  (register on import)
