from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import torch.nn as nn
from torch import Tensor


class Embeddings(NamedTuple):
    cls: Tensor | None
    reg: Tensor | None
    patch: Tensor | None


class Model(nn.Module):
    def forward(self, batch: dict[str, Tensor]) -> Embeddings: ...


ModelFn = Callable[..., Model]
