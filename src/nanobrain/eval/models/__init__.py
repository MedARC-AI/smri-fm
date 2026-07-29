from nanobrain.eval.utils import make_registry

register_model, create_model, list_models = make_registry()

from nanobrain.eval.models import (  # noqa: E402,F401  (register on import)
    neurojepa,
    random_features,
    unet,
)
