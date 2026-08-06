"""Random-projection baseline: a fixed, untrained backbone.

A sanity floor for real backbones and a smoke test for the harness. `global_embed` resizes
the volume to a cube and projects the flattened voxels; `patch_embed` projects each `patch^3`
block on the RAS-canonical grid and reports the block's world-mm centre. Patch features get
their own narrower `patch_dim`, since one vector per block is the memory cost.
"""

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from nibabel.affines import apply_affine
from torch import Tensor

from nanobrain.eval.models import register_model
from nanobrain.eval.models.base import PatchFeatures
from nanobrain.eval.nifti import canonical, canonical_img, normalize, resize


class RandomFeatures(nn.Module):
    def __init__(self, size: int = 64, patch: int = 16, dim: int = 1024, patch_dim: int = 128):
        super().__init__()
        self.size = size
        self.patch = patch
        self.global_proj = nn.Parameter(projection(size**3, dim), requires_grad=False)
        self.patch_proj = nn.Parameter(projection(patch**3, patch_dim), requires_grad=False)

    @torch.inference_mode()
    def global_embed(self, img: nib.Nifti1Image) -> Tensor:
        data = normalize(resize(canonical(img), self.size))  # (S, S, S)
        return data.flatten().to(self.global_proj.device) @ self.global_proj  # (D,)

    @torch.inference_mode()
    def patch_embed(self, img: nib.Nifti1Image) -> PatchFeatures:
        data = normalize(canonical(img))  # (X, Y, Z)
        p = self.patch
        pad = [(p - s % p) % p for s in data.shape]  # pad each axis up to a multiple of patch
        data = F.pad(data, (0, pad[2], 0, pad[1], 0, pad[0]))
        blocks = rearrange(data, "(x px) (y py) (z pz) -> x y z (px py pz)", px=p, py=p, pz=p)
        emb = blocks.to(self.patch_proj.device) @ self.patch_proj  # (nx, ny, nz, D)

        # same "x y z ->" ordering as the features, so row i of each describes the same block
        centres = rearrange(np.indices(emb.shape[:3]), "c x y z -> (x y z) c") * p + (p - 1) / 2
        coords = apply_affine(canonical_img(img).affine, centres)
        return PatchFeatures(
            rearrange(emb, "x y z d -> (x y z) d").cpu(),
            torch.from_numpy(coords).float(),
        )


def projection(in_dim: int, out_dim: int) -> Tensor:
    return torch.randn(in_dim, out_dim) / in_dim**0.5


@register_model
def random_features(
    size: int = 64, patch: int = 16, dim: int = 1024, patch_dim: int = 128
) -> RandomFeatures:
    return RandomFeatures(size=size, patch=patch, dim=dim, patch_dim=patch_dim)
