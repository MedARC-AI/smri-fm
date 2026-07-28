"""Random-weight 3D U-Net: a fixed, untrained convolutional backbone.

A second dumb baseline alongside `random_features`: same frozen-feature protocol, but the fixed
projection is a shallow encoder-decoder CNN, so features are local, multi-scale and
translation-equivariant. `global_embed` runs the encoder on a resized cube; `dense_embed` runs the
full U-Net at native resolution, upsampling onto each skip's grid so the features land back on the
input grid exactly.

Weights are torch's default init, frozen. InstanceNorm after every conv keeps activations in scale
without training, but it also fixes each channel's volume-wide mean, so a plain global average
pool is near-constant across subjects. `global_embed` instead average-pools every encoder stage
onto a coarse `pool^3` grid and concatenates: keeping regional structure and the shallow stages'
texture is worth a lot (300 ADNI subjects, brain-age Pearson r: 0.04 pooled to a single vector,
0.12 bottleneck on a 2^3 grid, 0.30 all stages on a 2^3 grid).
"""

import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from nanobrain.eval.models import register_model
from nanobrain.eval.nifti import canonical, normalize, resize


class RandomUNet(nn.Module):
    """`levels` stages, channels doubling with depth from `base`, halving resolution each stage."""

    def __init__(self, size: int = 96, base: int = 32, levels: int = 4, pool: int = 2):
        super().__init__()
        widths = [base * 2**level for level in range(levels)]
        self.size = size
        self.pool = pool
        self.encoder = nn.ModuleList(
            [_block(1, widths[0])]
            + [_block(shallow, deep, stride=2) for shallow, deep in zip(widths, widths[1:])]
        )
        self.decoder = nn.ModuleList(
            [_block(deep + shallow, shallow) for shallow, deep in zip(widths, widths[1:])]
        )
        self.requires_grad_(False)

    @torch.inference_mode()
    def global_embed(self, img: nib.Nifti1Image) -> Tensor:
        volume = resize(canonical(img), self.size)
        stages = self._encode(normalize(volume))
        pooled = [F.adaptive_avg_pool3d(stage, self.pool).flatten() for stage in stages]
        return torch.cat(pooled)  # (sum(widths) * pool^3,)

    @torch.inference_mode()
    def dense_embed(self, img: nib.Nifti1Image) -> Tensor:
        stages = self._encode(normalize(canonical(img)))
        dense = self._decode(stages)[0]  # (base, X, Y, Z)
        return dense.permute(1, 2, 3, 0).contiguous().cpu()  # (X, Y, Z, base)

    def _encode(self, volume: Tensor) -> list[Tensor]:
        """One (1, C, X, Y, Z) feature map per stage, coarsest last."""
        device = next(self.parameters()).device
        x = volume[None, None].to(device)
        stages = []
        for block in self.encoder:
            x = block(x)
            stages.append(x)
        return stages

    def _decode(self, stages: list[Tensor]) -> Tensor:
        """Upsample to each skip's grid and fuse, back up to the input resolution."""
        x = stages[-1]
        for block, skip in zip(reversed(self.decoder), reversed(stages[:-1])):
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
            x = block(torch.cat([x, skip], dim=1))
        return x


def _block(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
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
def random_unet(size: int = 96, base: int = 32, levels: int = 4, pool: int = 2) -> RandomUNet:
    return RandomUNet(size=size, base=base, levels=levels, pool=pool)
