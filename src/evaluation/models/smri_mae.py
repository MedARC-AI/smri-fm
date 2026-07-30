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
        pad_to_multiple: int | None = 32,
    ):
        super().__init__()
        self.encoder = encoder
        self.global_pool = global_pool
        self.pad_to_multiple = pad_to_multiple

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        images = batch["image"]
        mask = batch["mask"]

        cls, reg, patch, _, _, token_mask = self.encoder(
            images,
            mask=mask,
            pad_to_multiple=self.pad_to_multiple,
        )

        if self.global_pool == "cls":
            embed = cls[:, 0, :]
        elif self.global_pool == "reg":
            embed = reg.mean(dim=1)
        elif self.global_pool == "patch":
            token_mask = token_mask.to(device=patch.device, dtype=torch.bool)
            denom = token_mask.sum(dim=1, keepdim=True).clamp(min=1).to(dtype=patch.dtype)
            embed = (patch * token_mask.unsqueeze(-1)).sum(dim=1) / denom
        return embed


class SmriMaeTransform:
    def __init__(
        self,
        img_size: tuple[int, int, int],
        spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ):
        self.img_size = img_size
        self.spacing = spacing

    def __call__(self, img: nib.Nifti1Image) -> dict[str, Tensor]:
        """
        TODO(mihir): check
        """
        # reorient to RAS
        img = nib.as_closest_canonical(img)

        # note, shape is (X, Y, Z) in contiguous F-order
        data = img.get_fdata(dtype=np.float32)
        data = torch.from_numpy(data)
        spacing = img.header.get_zooms()

        # resize
        if max(abs(s - s_) for s, s_ in zip(spacing, self.spacing)) > 0.05:
            data = rescale(data, spacing, target_spacing=self.spacing)

        # Preserve the canonical NIfTI (X, Y, Z) axis order used in pretraining.
        data = data.contiguous()
        data = pad_to_shape(data, self.img_size)

        # cheap mask
        # if we want a better mask, we have to compute it here.
        # model contract is nifti image -> embedding
        mask = data > data.mean()

        # z-score over brain-mask voxels (matches pretraining); background -> 0.
        # Raw intensities reach ~1e6, so this must happen before the fp16 cast.
        brain = data[mask]
        # population std (÷N, correction=0) to match the pretraining normalization.
        mean, std = brain.mean(), brain.std(correction=0).clamp_min(1e-6)
        data = torch.where(mask, (data - mean) / std, 0.0)

        # fp16 and add channel dim
        data = data.half().unsqueeze(0)
        mask = mask.unsqueeze(0)

        sample = {"image": data, "mask": mask}
        return sample


# can copy these utils to shared module if they prove generally useful


def rescale(
    x: torch.Tensor,
    spacing: tuple[float, ...],
    target_spacing: tuple[float, ...] = (1.0, 1.0, 1.0),
):
    scales = tuple([current / target for current, target in zip(spacing, target_spacing)])
    x = F.interpolate(x[None, None], scale_factor=scales, mode="trilinear").squeeze(0, 1)
    return x


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

    backbone = SmriMaeBackbone(
        model.encoder,
        global_pool=global_pool,
        pad_to_multiple=args.get("pad_to_multiple", 32),
    )
    transform = SmriMaeTransform(img_size=args["img_size"])

    return backbone, transform
