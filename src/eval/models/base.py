from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import nibabel as nib
import torch.nn as nn
from torch import Tensor


class Embeddings(NamedTuple):
    cls: Tensor | None
    reg: Tensor | None
    patch: Tensor | None


class Model(nn.Module):
    def forward(self, batch: dict[str, Tensor]) -> Embeddings: ...


class Transform:
    def __call__(self, img: nib.Nifti1Image) -> dict[str, Tensor]: ...


ModelTransformPair = tuple[Transform | None, Model]
ModelFn = Callable[..., Model | ModelTransformPair]
