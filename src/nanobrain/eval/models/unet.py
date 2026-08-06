"""Random-weight 3D U-Net: a fixed, untrained convolutional baseline next to `random_features`."""

import math

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


class RandomUNet(nn.Module):
    """`levels` stages, channels doubling with depth from `base`, halving resolution each stage.

    `global_embed` takes the centre cell of a `pool`-cubed pooling of the deepest stage only, so
    the probe sees a few hundred features rather than a few thousand. `patch_embed` decodes back
    up to a stride of `patch` voxels rather than to full resolution, which is the scale the real
    backbones work at anyway.
    """

    def __init__(
        self, size: int = 96, base: int = 32, levels: int = 4, pool: int = 3, patch: int = 4
    ):
        super().__init__()
        widths = [base * 2**level for level in range(levels)]
        self.decode_level = int(math.log2(patch))
        assert 2**self.decode_level == patch, f"patch must be a power of two, got {patch}"
        assert self.decode_level < levels - 1, (
            f"patch must be under {2 ** (levels - 1)} to decode at all"
        )
        self.size = size
        self.pool = pool
        self.patch = patch
        self.encoder = nn.ModuleList(
            [conv_block(1, widths[0])]
            + [conv_block(shallow, deep, stride=2) for shallow, deep in zip(widths, widths[1:])]
        )
        self.decoder = nn.ModuleList(
            [conv_block(deep + shallow, shallow) for shallow, deep in zip(widths, widths[1:])]
        )
        self.requires_grad_(False)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @torch.inference_mode()
    def global_embed(self, img: nib.Nifti1Image) -> Tensor:
        data = normalize(resize(canonical(img), self.size))  # (S, S, S)
        deepest = self.encode(data.to(self.device))[-1]
        pooled = F.adaptive_avg_pool3d(deepest, self.pool)  # (1, C, P, P, P)
        centre = self.pool // 2
        return pooled[0, :, centre, centre, centre]  # (base * 2^(levels-1),)

    @torch.inference_mode()
    def patch_embed(self, img: nib.Nifti1Image) -> PatchFeatures:
        data = normalize(canonical(img))  # (X, Y, Z)
        stages = self.encode(data.to(self.device))
        dense = self.decode(stages)[0]  # (base * patch, X/patch, Y/patch, Z/patch)

        # stride-2 padded convs keep cell j centred on input voxel j * patch; the "x y z ->"
        # ordering matches the features, so row i of each describes the same cell
        centres = rearrange(np.indices(dense.shape[1:]), "c x y z -> (x y z) c") * self.patch
        coords = apply_affine(canonical_img(img).affine, centres)
        return PatchFeatures(
            rearrange(dense, "c x y z -> (x y z) c").cpu(),
            torch.from_numpy(coords).float(),
        )

    def encode(self, volume: Tensor) -> list[Tensor]:
        """One (1, C, X, Y, Z) feature map per stage, coarsest last."""
        x = volume[None, None]
        stages = []
        for block in self.encoder:
            x = block(x)
            stages.append(x)
        return stages

    def decode(self, stages: list[Tensor]) -> Tensor:
        """Upsample to each skip's grid and fuse, deepest stage back up to `self.decode_level`."""
        x = stages[-1]
        for block, skip in zip(
            reversed(self.decoder[self.decode_level :]), reversed(stages[self.decode_level : -1])
        ):
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
            x = block(torch.cat([x, skip], dim=1))
        return x


def conv_block(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    """Two 3x3x3 convs, the first optionally striding down."""
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1),
        nn.InstanceNorm3d(out_ch),
        nn.LeakyReLU(0.01, inplace=True),
        nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.InstanceNorm3d(out_ch),
        nn.LeakyReLU(0.01, inplace=True),
    )


@register_model
def random_unet(
    size: int = 96, base: int = 32, levels: int = 4, pool: int = 3, patch: int = 4
) -> RandomUNet:
    return RandomUNet(size=size, base=base, levels=levels, pool=pool, patch=patch)
