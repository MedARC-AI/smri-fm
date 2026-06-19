from typing import Literal

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from evaluation.models.registry import register_model
from evaluation.utils import load_mni_brain_mask, resample_binary_mask

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
    def __init__(
        self,
        img_size: tuple[int, int, int],
        spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ):
        self.img_size = img_size
        self.spacing = spacing

    def __call__(self, img: nib.Nifti1Image) -> dict[str, Tensor]:
        # reorient to RAS
        img = nib.as_closest_canonical(img)

        # note, shape is (X, Y, Z) in contiguous F-order
        data_np = img.get_fdata(dtype=np.float32)
        spacing = tuple(float(z) for z in img.header.get_zooms()[:3])

        # brain mask = the MNI brain template resampled onto this subject's grid
        # (nearest-neighbor, alignment via affines). The scans are MNI-registered, so the
        # template prior alone agrees well with SynthSeg (dice ~0.85 on ADNI).
        mask_np = resample_binary_mask(load_mni_brain_mask(), img)

        data = torch.from_numpy(data_np)
        mask = torch.from_numpy(mask_np)

        # resize to target spacing (image: trilinear, mask: nearest to stay binary)
        if max(abs(s - s_) for s, s_ in zip(spacing, self.spacing)) > 0.05:
            data = rescale(data, spacing, target_spacing=self.spacing)
            mask = rescale_mask(mask, spacing, target_spacing=self.spacing)

        # tranpose (X, Y, Z) F-order -> (Z, Y, X) C-order
        # TODO: this shape issue is a footgun. need to be consistent and obvious about
        # whether we are doing (X, Y, Z) or (Z, Y, X) for image as well as img_size,
        # spacing.
        data = data.permute(2, 1, 0).contiguous()
        mask = mask.permute(2, 1, 0).contiguous()

        # z-score over brain-mask voxels (matches pretraining); background -> 0.
        # Raw intensities reach ~1e6, so this must happen before the low-precision cast.
        brain = data[mask]
        # population std (÷N, correction=0) to match the pretraining normalization.
        mean, std = brain.mean(), brain.std(correction=0).clamp_min(1e-6)
        data = torch.where(mask, (data - mean) / std, 0.0)

        # pad/crop both image and mask to the model input size
        data = pad_to_shape(data, self.img_size)
        mask = pad_to_shape(mask.to(torch.int8), self.img_size)

        # fp16 image, int8 mask (1=brain, 0=background); add channel dim
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


def rescale_mask(
    mask: torch.Tensor,
    spacing: tuple[float, ...],
    target_spacing: tuple[float, ...] = (1.0, 1.0, 1.0),
):
    # nearest-neighbor so the mask stays binary; matches `rescale`'s grid exactly.
    scales = tuple([current / target for current, target in zip(spacing, target_spacing)])
    out = F.interpolate(mask[None, None].float(), scale_factor=scales, mode="nearest")
    return out.squeeze(0, 1) > 0.5


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
