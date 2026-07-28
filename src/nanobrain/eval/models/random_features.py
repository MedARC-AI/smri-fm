"""Random-projection baseline: a fixed, untrained backbone.

A sanity floor for real backbones and a smoke test for the harness. `global_embed` resizes
the volume to a cube and projects the flattened voxels; `dense_embed` projects each `patch^3`
block on the RAS-canonical grid and broadcasts its embedding to that block's voxels. Dense
features get their own narrower `patch_dim`, since one vector per voxel is the memory cost.
"""

import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from nanobrain.eval.models import register_model
from nanobrain.eval.nifti import canonical, normalize, resize


class RandomFeatures(nn.Module):
    def __init__(self, size: int = 64, patch: int = 16, dim: int = 1024, patch_dim: int = 128):
        super().__init__()
        self.size = size
        self.patch = patch
        self.global_proj = nn.Parameter(_projection(size**3, dim), requires_grad=False)
        self.patch_proj = nn.Parameter(_projection(patch**3, patch_dim), requires_grad=False)

    @torch.inference_mode()
    def global_embed(self, img: nib.Nifti1Image) -> Tensor:
        data = normalize(resize(canonical(img), self.size))  # (S, S, S)
        return data.flatten().to(self.global_proj.device) @ self.global_proj  # (D,)

    @torch.inference_mode()
    def dense_embed(self, img: nib.Nifti1Image) -> Tensor:
        data = normalize(canonical(img))  # (X, Y, Z)
        shape = data.shape
        p = self.patch
        pad = [(p - s % p) % p for s in shape]  # pad each axis up to a multiple of patch
        data = F.pad(data, (0, pad[2], 0, pad[1], 0, pad[0]))
        blocks = rearrange(data, "(x px) (y py) (z pz) -> x y z (px py pz)", px=p, py=p, pz=p)
        emb = blocks.to(self.patch_proj.device) @ self.patch_proj  # (nx, ny, nz, D)
        emb = emb.repeat_interleave(p, 0).repeat_interleave(p, 1).repeat_interleave(p, 2)
        return emb[: shape[0], : shape[1], : shape[2]].contiguous().cpu()  # (X, Y, Z, D)


def _projection(in_dim: int, out_dim: int) -> Tensor:
    return torch.randn(in_dim, out_dim) / in_dim**0.5


@register_model
def random_features(
    size: int = 64, patch: int = 16, dim: int = 1024, patch_dim: int = 128
) -> RandomFeatures:
    return RandomFeatures(size=size, patch=patch, dim=dim, patch_dim=patch_dim)
