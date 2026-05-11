from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class DummyBackbone(nn.Module):
    """Small deterministic frozen backbone for smoke-testing probe evaluations."""

    def __init__(
        self,
        embedding_dim: int = 8,
        input_shape: tuple[int, int, int, int] | list[int] = (1, 8, 8, 8),
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.input_shape = tuple(input_shape)
        input_dim = math.prod(self.input_shape)

        projection = torch.linspace(-0.5, 0.5, steps=input_dim * embedding_dim)
        self.projection = nn.Linear(input_dim, embedding_dim, bias=True)
        with torch.no_grad():
            self.projection.weight.copy_(projection.reshape(input_dim, embedding_dim).T)
            self.projection.bias.zero_()
        self.requires_grad_(False)

    def forward(self, batch: Tensor) -> Tensor:
        return self.encode(batch)

    def encode(self, batch: Tensor) -> Tensor:
        if batch.ndim != len(self.input_shape) + 1:
            raise ValueError(
                f"Expected batch shape [B, {', '.join(map(str, self.input_shape))}], "
                f"got {tuple(batch.shape)}."
            )
        return self.projection(batch.flatten(start_dim=1))
