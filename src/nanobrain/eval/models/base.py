"""The model contract every eval backbone implements.

Each model takes one nifti and canonicalizes/normalizes it internally, exposing two views:
- `global_embed`: one pooled vector per volume (classification / regression).
- `dense_embed`: one feature vector per voxel on the RAS-canonical grid (segmentation), on CPU.

Backbones are `nn.Module`s so the harness can `.to(device)` / `.eval()`; each method then moves
its own inputs onto the module's device. Both views resolve on the RAS-canonical grid, which is
the shared contract the segmentation probe aligns its labels to. The task owns the sampling grid,
resampling image + seg to a per-task spacing before the model sees them.
"""

from typing import Protocol

import nibabel as nib
from torch import Tensor


class Model(Protocol):
    def global_embed(self, img: nib.Nifti1Image) -> Tensor:
        """(D,) one pooled embedding for the whole volume."""
        ...

    def dense_embed(self, img: nib.Nifti1Image) -> Tensor:
        """(X, Y, Z, D) one embedding per voxel on the RAS-canonical grid, on CPU."""
        ...
