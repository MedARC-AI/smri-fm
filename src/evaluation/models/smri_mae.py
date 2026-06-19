from typing import Literal

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from evaluation.models.registry import register_model

import smri_mae.model_mae as models_mae


class SmriMaeBackbone(nn.Module):
    def __init__(
        self,
        encoder: models_mae.MaskedEncoder,
        global_pool: Literal["cls", "reg", "patch"] = "patch",
    ):
        super().__init__()
        self.encoder = encoder
        self.global_pool = global_pool

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        images = batch["image"]
        mask = batch["mask"]

        cls, reg, patch = self.encoder.forward_embedding(images, mask=mask)

        if self.global_pool == "cls":
            embed = cls[:, 0, :]
        elif self.global_pool == "reg":
            embed = reg.mean(dim=1)
        elif self.global_pool == "patch":
            embed = patch.mean(dim=1)
        return embed


class SmriMaeTransform:
    def __init__(self, img_size: tuple[int, int, int]):
        self.img_size = img_size

    def __call__(self, img: nib.Nifti1Image, mask: nib.Nifti1Image) -> dict[str, Tensor]:
        # the dataset ships preprocessed volumes (RAS, isotropic) and a brain mask,
        # so the transform only normalizes, pads, and casts to fp16.
        data = torch.from_numpy(img.get_fdata(dtype=np.float32))
        mask = torch.from_numpy(mask.get_fdata(dtype=np.float32))

        # pad/crop to model input size
        data = pad_to_shape(data, self.img_size)
        mask = pad_to_shape(mask, self.img_size) > 0.5

        # z-score over brain-mask voxels (matches pretraining); background -> 0.
        # Raw intensities reach ~1e6, so this must happen before the low-precision cast.
        brain = data[mask]
        # population std (÷N, correction=0) to match the pretraining normalization.
        mean, std = brain.mean(), brain.std(correction=0).clamp_min(1e-6)
        data = torch.where(mask, (data - mean) / std, 0.0)

        # fp16 and add channel dim
        data = data.half().unsqueeze(0)
        mask = mask.unsqueeze(0)

        sample = {"image": data, "mask": mask}
        return sample


def pad_to_shape(x: torch.Tensor, target_shape: tuple[int, ...]):
    # nb this also crops
    padding = []
    for s, s_ in reversed(list(zip(x.shape, target_shape))):
        pad = s_ - s
        padding.extend([pad // 2, pad - pad // 2])
    x = F.pad(x, padding)
    return x


@register_model
def smri_mae(ckpt_path: str, global_pool: str = "patch"):
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

    backbone = SmriMaeBackbone(model.encoder, global_pool=global_pool)
    transform = SmriMaeTransform(img_size=args["img_size"])

    return backbone, transform
