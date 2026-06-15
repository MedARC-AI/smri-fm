import torch
import torch.nn as nn
from torch import Tensor

from eval.models.base import Embeddings
from eval.models.registry import register_model

import smri_mae.model_mae as models_mae


class SMRIMAEBackbone(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, batch: dict[str, Tensor]) -> Embeddings:
        cls, reg, patch = self.encoder.forward_embedding(batch["image"], mask=batch["mask"])
        cls = cls[:, 0, :] if cls is not None else None
        return Embeddings(cls=cls, reg=reg, patch=patch)


@register_model
def smri_mae(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    args = ckpt["args"]

    model_fn = models_mae.__dict__[args["model"]]
    model: models_mae.MaskedAutoencoderViT = model_fn(
        img_size=args["img_size"],
        in_chans=args.get("in_chans", 1),
        patch_size=args["patch_size"],
        **(args.get("model_kwargs") or {}),
    )
    model.load_state_dict(ckpt["model"])

    return SMRIMAEBackbone(model.encoder)
