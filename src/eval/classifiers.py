from __future__ import annotations

import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F

from eval.utils import select_representation


class ClassifierGrid(nn.Module):
    """Runs a frozen backbone once and applies a sweep of classifier heads.

    Each head corresponds to one (lr_multiplier, wd_multiplier) hyperparameter
    combination. forward returns stacked logits of shape
    [batch, out_dim, num_classifiers] so every head is trained/evaluated together.
    """

    def __init__(self, backbone: nn.Module, representation: str,
                 classifiers: dict[tuple[float, float], nn.Module]):
        super().__init__()
        self.backbone = backbone
        self.representation = representation
        # ModuleDict keys must be dot-free strings, so track hparams separately.
        self.hparams = list(classifiers)
        self.classifiers = nn.ModuleList(classifiers.values())

    def forward(self, batch) -> torch.Tensor:
        embeds = select_representation(self.backbone(batch), self.representation)
        return torch.stack([clf(embeds) for clf in self.classifiers], dim=-1)


class LinearClassifier(nn.Module):
    def __init__(self, in_dim, out_dim, norm=False, xavier_init=True):
        super().__init__()
        self.norm = nn.BatchNorm1d(in_dim, affine=False) if norm else nn.Identity()
        self.linear = nn.Linear(in_dim, out_dim)
        if xavier_init:
            nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        if x.ndim == 3:
            x = x.mean(dim=1)
        return self.linear(self.norm(x))


class MLPClassifier(nn.Module):
    def __init__(self, in_dim, out_dim, embed_dim=None, dropout=0.0):
        super().__init__()
        embed_dim = embed_dim or in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, embed_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, out_dim),
        )

    def forward(self, x):
        if x.ndim == 3:
            x = x.mean(dim=1)
        return self.net(x)


class AttnPoolClassifier(nn.Module):
    def __init__(self, in_dim, out_dim, embed_dim=None):
        super().__init__()
        embed_dim = embed_dim or in_dim
        embed_dim = max(64, 64 * (embed_dim // 64))
        self.query = nn.Parameter(torch.empty(1, 1, embed_dim))
        self.kv = nn.Linear(in_dim, embed_dim * 2)
        self.head = nn.Linear(embed_dim, out_dim)
        self.num_heads = embed_dim // 64
        nn.init.trunc_normal_(self.query, std=0.02)

    def forward(self, x):
        if x.ndim == 2:
            x = x[:, None]
        batch, length, _ = x.shape
        dim = self.query.shape[-1]
        q = self.query.expand(batch, -1, -1).reshape(batch, 1, self.num_heads, 64).transpose(1, 2)
        kv = self.kv(x).reshape(batch, length, 2, self.num_heads, 64).permute(2, 0, 3, 1, 4)
        pooled = F.scaled_dot_product_attention(q, kv[0], kv[1]).reshape(batch, dim)
        return self.head(pooled)


CLASSIFIERS = {"linear": LinearClassifier, "mlp": MLPClassifier, "attn": AttnPoolClassifier}


def create_classifier(name, in_dim, out_dim, **kwargs):
    cls = CLASSIFIERS[name]
    params = inspect.signature(cls).parameters
    return cls(in_dim=in_dim, out_dim=out_dim, **{k: v for k, v in kwargs.items() if k in params})


def list_classifiers():
    return sorted(CLASSIFIERS)
