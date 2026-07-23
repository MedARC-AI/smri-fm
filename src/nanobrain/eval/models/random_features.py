"""Random-projection baseline: a fixed, untrained backbone.

Resamples every volume to a cube, then projects flattened voxels (global) or flattened
patches (dense) through frozen random matrices. A sanity floor for real backbones and a
cheap way to smoke-test the whole harness end to end.
"""

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from nanobrain.eval.models import register_model
from nanobrain.eval.models.base import ModelTransform


class RandomFeaturesTransform:
    def __init__(self, size: int):
        self.size = size

    def __call__(
        self, img: nib.Nifti1Image, seg: nib.Nifti1Image | None = None
    ) -> dict[str, Tensor]:
        data = torch.from_numpy(nib.as_closest_canonical(img).get_fdata(dtype=np.float32))
        data = resize(data, self.size)
        brain = data > data.mean()
        mean = data[brain].mean()
        std = data[brain].std().clamp_min(1e-6)
        sample = {"image": torch.where(brain, (data - mean) / std, 0.0).unsqueeze(0)}
        if seg is not None:
            seg_data = torch.from_numpy(nib.as_closest_canonical(seg).get_fdata(dtype=np.float32))
            sample["seg"] = resize(seg_data, self.size, nearest=True).unsqueeze(0)
        return sample  # image, seg: (1, S, S, S)


class RandomFeatures(nn.Module):
    def __init__(self, size: int = 64, patch: int = 16, dim: int = 1024, seed: int = 0):
        super().__init__()
        assert size % patch == 0, f"size {size} must be divisible by patch {patch}"
        self.size = size
        self.patch = patch
        generator = torch.Generator().manual_seed(seed)
        self.global_proj = nn.Parameter(_projection(size**3, dim, generator), requires_grad=False)
        self.patch_proj = nn.Parameter(_projection(patch**3, dim, generator), requires_grad=False)

    @torch.inference_mode()
    def global_embed(self, batch: dict[str, Tensor]) -> Tensor:
        voxels = batch["image"].flatten(1)  # (B, S^3)
        return voxels @ self.global_proj

    @torch.inference_mode()
    def patch_embed(self, batch: dict[str, Tensor]) -> Tensor:
        patches = self._to_patches(batch["image"])  # (B, N, patch^3)
        return patches @ self.patch_proj

    @torch.inference_mode()
    def patchify_labels(self, seg: Tensor) -> Tensor:
        patches = self._to_patches(seg.unsqueeze(0))  # (1, S, S, S) -> (1, N, patch^3)
        return (patches > 0).float().mean(dim=-1).squeeze(0)  # (N,)

    def _to_patches(self, volume: Tensor) -> Tensor:
        p = self.patch
        return rearrange(
            volume, "b c (x px) (y py) (z pz) -> b (x y z) (c px py pz)", px=p, py=p, pz=p
        )


def _projection(in_dim: int, out_dim: int, generator: torch.Generator) -> Tensor:
    return torch.randn(in_dim, out_dim, generator=generator) / in_dim**0.5


def resize(volume: Tensor, size: int, nearest: bool = False) -> Tensor:
    mode = "nearest" if nearest else "trilinear"
    kwargs = {} if nearest else {"align_corners": False}
    resized = F.interpolate(volume[None, None], size=(size, size, size), mode=mode, **kwargs)
    return resized[0, 0]


@register_model
def random_features(
    size: int = 64, patch: int = 16, dim: int = 1024, seed: int = 0
) -> ModelTransform:
    model = RandomFeatures(size=size, patch=patch, dim=dim, seed=seed)
    transform = RandomFeaturesTransform(size=size)
    return model, transform
