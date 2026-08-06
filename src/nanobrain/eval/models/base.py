"""The model contract every eval backbone implements.

Each model takes one nifti and canonicalizes/normalizes it internally, exposing two views:
- `global_embed`: one pooled vector per volume (classification / regression).
- `patch_embed`: localized patch features plus their world coordinates (segmentation), on CPU.

Backbones are `nn.Module`s so the harness can `.to(device)` / `.eval()`; each method then moves
its own inputs onto the module's device. Patch coordinates are RAS world millimetres in the input
image's frame, so the segmentation probe can match features to label voxels without the model ever
seeing the label grid: every backbone is scored on the task's own grid whatever its patch size.
"""

from typing import NamedTuple, Protocol

import nibabel as nib
from torch import Tensor


class PatchFeatures(NamedTuple):
    features: Tensor  # (N, D)
    coords: Tensor  # (N, 3) patch centres, RAS world mm in the input image's frame


class Model(Protocol):
    def global_embed(self, img: nib.Nifti1Image) -> Tensor:
        """(D,) one pooled embedding for the whole volume."""
        ...

    def patch_embed(self, img: nib.Nifti1Image) -> PatchFeatures:
        """`N` patch embeddings and the world-mm centre of each, on CPU."""
        ...
