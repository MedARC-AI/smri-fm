"""Asparagus-compatible wrappers around the smri_mae MAE backbone.

These satisfy the minimal interface asparagus' finetune pipeline expects:
    - constructor accepts (input_channels, output_channels, ...) kwargs
    - exposes a `num_classes` attribute (set from output_channels)
    - forward(x) -> logits of the appropriate shape for the task
    - weights loaded later by asparagus.BaseModule via load_state_dict(strict=False)
"""

import torch.nn as nn
from torch import Tensor

from smri_mae.model_mae import MaskedViT


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
        cls_embeds, _, patch_embeds, _, _ = self.encoder(x)
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


class SmriMaeSegBackbone(nn.Module):
    """Placeholder for ViT-based segmentation backbone."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("SmriMaeSegBackbone is not yet implemented")

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError
