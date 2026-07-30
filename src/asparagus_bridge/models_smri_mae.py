"""Asparagus-compatible wrappers around the smri_mae MAE backbone.

These satisfy the minimal interface asparagus' finetune pipeline expects:
    - constructor accepts (input_channels, output_channels, ...) kwargs
    - exposes a `num_classes` attribute (set from output_channels)
    - forward(x) -> logits of the appropriate shape for the task
    - weights loaded later by asparagus.BaseModule via load_state_dict(strict=False)
"""

import math

from gardening_tools.modules.networks.BaseNet import BaseNet
from gardening_tools.modules.networks.components.transformer import PatchDecode
from gardening_tools.modules.networks.utils import get_steps_for_sliding_window
import torch
import torch.nn as nn
from torch import Tensor

from smri_mae.model_mae import MaskedViT


def gaussian_window(patch_size: tuple[int, ...], sigma_scale: float = 0.125) -> Tensor:
    """Separable Gaussian weight map over a sliding-window patch.

    Peaks at the patch centre and decays toward the borders, so a voxel is
    scored mainly by the windows that saw it with the most spatial context.
    Same construction (and default sigma) nnU-Net uses for its window blending.
    """
    axes = []
    for size in patch_size:
        coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
        axes.append(torch.exp(-0.5 * (coords / (size * sigma_scale)) ** 2))

    window = axes[0]
    for axis in axes[1:]:
        window = window[..., None] * axis
    # Floor at a small positive value: a zero-weight voxel would divide by zero
    # wherever it is the only window covering that voxel.
    return torch.clamp(window / window.max(), min=1e-4)


class SmriMaeClsRegBackbone(nn.Module):
    """MAE ViT classifier/regressor for asparagus cls + reg downstream tasks.

    Single class serves both because the architecture is identical — only the
    upstream loss differs (CE for cls, L1/MSE for reg).
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        img_size: int | tuple[int, int, int] = (160, 160, 160),
        patch_size: int | tuple[int, int, int] = (16, 16, 16),
        depth: int = 12,
        embed_dim: int = 768,
        num_heads: int = 12,
        pool: str = "cls",
        dimensions: str = "3D",
        **_ignored,
    ):
        super().__init__()
        assert dimensions == "3D", f"only 3D supported, got dimensions={dimensions}"
        assert pool in {"cls", "mean"}, f"pool must be 'cls' or 'mean', got {pool}"

        self.num_classes = output_channels
        self.pool = pool

        self.encoder = MaskedViT(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=input_channels,
            depth=depth,
            embed_dim=embed_dim,
            num_heads=num_heads,
            class_token=(pool == "cls"),
        )
        self.head = nn.Linear(embed_dim, output_channels)

    def _features(self, x: Tensor) -> Tensor:
        """Encoder output"""
        cls_embeds, _, patch_embeds, _, _, _ = self.encoder(x)
        if self.pool == "cls":
            return cls_embeds.squeeze(1)
        else:
            return patch_embeds.mean(dim=1)

    def forward(self, x: Tensor) -> Tensor:
        """Encoder + head """
        return self.head(self._features(x))

    def _encode(self, x: Tensor) -> Tensor:
        """ Encoder output in format used for linear probing"""
        feat = self._features(x)
        return feat[:, :, None, None, None]


class LinearPatchDecode(nn.Module):
    """One linear layer from patch tokens to voxel logits.

    The DINOv2 segmentation-eval protocol: no convolutions, no upsampling
    stages, no learned spatial mixing. Each token is projected straight to the
    logits of the voxels inside its own patch, so a voxel's prediction depends
    only on the token covering it. Whatever it scores is what the frozen
    representation already carries, rather than what a decoder can reconstruct
    on top of it.

    Deliberately weaker than `PatchDecode` and blocky at patch boundaries. That
    is the point of a probe.
    """

    def __init__(self, patch_size, embed_dim: int, out_channels: int):
        super().__init__()
        self.patch_size = tuple(patch_size)
        self.num_classes = out_channels
        self.proj = nn.Linear(embed_dim, math.prod(self.patch_size) * out_channels)

    def forward(self, x: Tensor) -> Tensor:
        # (B, embed_dim, gx, gy, gz) -> (B, out_channels, gx*px, gy*py, gz*pz)
        batch, _, gx, gy, gz = x.shape
        px, py, pz = self.patch_size

        tokens = x.permute(0, 2, 3, 4, 1)
        logits = self.proj(tokens)
        logits = logits.reshape(batch, gx, gy, gz, px, py, pz, self.num_classes)
        # interleave each grid axis with its within-patch axis before merging
        logits = logits.permute(0, 7, 1, 4, 2, 5, 3, 6)
        return logits.reshape(batch, self.num_classes, gx * px, gy * py, gz * pz)


class SmriMaeSegBackbone(BaseNet):
    """MAE ViT segmentation backbone with a Primus-like patch decoder."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        img_size: int | tuple[int, int, int] = (160, 160, 160),
        patch_size: int | tuple[int, int, int] = (16, 16, 16),
        depth: int = 12,
        embed_dim: int = 768,
        num_heads: int = 12,
        dimensions: str = "3D",
        head: str = "patch_decode",
        window_blending: str = "gaussian",
        **_ignored,
    ):
        super().__init__()
        assert dimensions == "3D", f"only 3D supported, got dimensions={dimensions}"

        self.num_classes = output_channels
        self.stem_weight_name = "encoder.patch_embed.weight"
        self.window_blending = window_blending

        self.encoder = MaskedViT(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=input_channels,
            depth=depth,
            embed_dim=embed_dim,
            num_heads=num_heads,
            class_token=True,
        )
        self.grid_size = self.encoder.patchify.grid_size
        if head == "patch_decode":
            self.decoder = PatchDecode(
                patch_size=self.encoder.patchify.patch_size,
                embed_dim=embed_dim,
                out_channels=output_channels,
            )
        elif head == "linear":
            self.decoder = LinearPatchDecode(
                patch_size=self.encoder.patchify.patch_size,
                embed_dim=embed_dim,
                out_channels=output_channels,
            )
        else:
            raise ValueError(
                f"head must be 'patch_decode' or 'linear', got {head!r}"
            )

    def forward(self, x: Tensor) -> Tensor:
        _, _, patch_embeds, _, _, _ = self.encoder(x)
        expected_tokens = math.prod(self.grid_size)
        if patch_embeds.shape[1] != expected_tokens:
            raise ValueError(
                "unexpected MAE patch token count: "
                f"got {patch_embeds.shape[1]}, expected {expected_tokens}"
            )

        features = patch_embeds.reshape(
            x.shape[0],
            *self.grid_size,
            patch_embeds.shape[-1],
        )
        features = features.permute(0, 4, 1, 2, 3).contiguous()
        return self.decoder(features)

    def _sliding_window_predict3D(self, data, patch_size, overlap):
        """Overlap-normalized sliding-window inference.

        BaseNet accumulates window logits into a canvas without dividing by the
        number of windows covering each voxel. Argmax at the inference
        resolution survives that, because a per-voxel scalar rescales every
        class equally. It does not survive what happens next: asparagus'
        `reverse_preprocessing` runs a trilinear `F.interpolate` on these raw
        logits to get back to source geometry, and interpolation is a weighted
        average *across* neighbouring voxels. Neighbours with different coverage
        counts then contribute in proportion to their inflated magnitude, so a
        voxel seen by 8 windows outvotes an adjacent one seen by 2 on window
        geometry alone.

        The counts are not close to uniform. For a 251x214x198 volume at
        patch=64, overlap=0.5 they span 1 to 27, because
        `get_steps_for_sliding_window` appends a final step at `shape - patch`
        regardless of stride, leaving a ragged tail window.
        """
        weight = self._window_weight(patch_size, data.device, data.dtype)

        logits = torch.zeros(
            (data.shape[0], self.num_classes, *data.shape[2:]),
            device=data.device,
            dtype=torch.float32,
        )
        coverage = torch.zeros(
            (1, 1, *data.shape[2:]), device=data.device, dtype=torch.float32
        )

        x_steps, y_steps, z_steps = get_steps_for_sliding_window(
            data.shape[2:], patch_size, overlap
        )
        px, py, pz = patch_size

        for xs in x_steps:
            for ys in y_steps:
                for zs in z_steps:
                    window = (slice(xs, xs + px), slice(ys, ys + py), slice(zs, zs + pz))
                    out = self.forward(data[(slice(None), slice(None), *window)])
                    logits[(slice(None), slice(None), *window)] += out.float() * weight
                    coverage[(slice(None), slice(None), *window)] += weight

        # Every voxel is covered by at least one window, so coverage is strictly
        # positive and the divide is safe.
        return logits / coverage

    def _window_weight(self, patch_size, device, dtype) -> Tensor:
        if self.window_blending == "gaussian":
            weight = gaussian_window(tuple(patch_size)).to(device=device)
        elif self.window_blending == "uniform":
            weight = torch.ones(tuple(patch_size), device=device)
        else:
            raise ValueError(
                "window_blending must be 'gaussian' or 'uniform', got "
                f"{self.window_blending!r}"
            )
        return weight
