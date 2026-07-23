"""The model contract every eval backbone implements.

A `Transform` preprocesses a nifti into a sample dict, co-transforming an optional
segmentation onto the image grid. The backbone exposes three views of that sample:
- `global_embed`: one pooled vector per volume (classification / regression).
- `patch_embed`: one token per patch (segmentation).
- `patchify_labels`: per-patch foreground fraction, on the same grid as `patch_embed`.

Backbones are `nn.Module`s; the harness calls `.to(device)` and `.eval()`.
"""

from typing import Protocol

import nibabel as nib
from torch import Tensor


class Transform(Protocol):
    def __call__(
        self, img: nib.Nifti1Image, seg: nib.Nifti1Image | None = None
    ) -> dict[str, Tensor]:
        """Preprocess `img`; if `seg` is given, resample it (nearest) onto the image grid
        and return it under "seg". Batch-collatable sample dict."""
        ...


class Model(Protocol):
    def global_embed(self, batch: dict[str, Tensor]) -> Tensor:
        """(B, D) one pooled embedding per volume."""
        ...

    def patch_embed(self, batch: dict[str, Tensor]) -> Tensor:
        """(B, N, D) one embedding per patch."""
        ...

    def patchify_labels(self, seg: Tensor) -> Tensor:
        """(N,) per-patch foreground fraction for an on-grid seg tensor from the transform."""
        ...


ModelTransform = tuple[Model, Transform]
