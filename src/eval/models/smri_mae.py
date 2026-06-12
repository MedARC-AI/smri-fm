from typing import Literal

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from eval.models.base import Embeddings
from eval.models.registry import register_model

import smri_mae.model_mae as models_mae


class SmriMaeBackbone(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, batch: dict[str, Tensor]) -> Embeddings:
        cls, reg, patch = self.encoder.forward_embedding(batch["image"], mask=batch["mask"])
        cls = cls[:, 0, :] if cls is not None else None
        return Embeddings(cls=cls, reg=reg, patch=patch)


class SmriMaeTransform:
    def __init__(
        self,
        img_size: tuple[int, int, int],
        spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ):
        self.img_size = img_size
        self.spacing = spacing

    def __call__(self, img: nib.Nifti1Image, brain_mask: nib.Nifti1Image | None = None) -> dict[str, Tensor]:
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
        data = rescale(data, spacing, target_spacing=self.spacing)

        # tranpose (X, Y, Z) F-order -> (Z, Y, X) C-order
        # TODO: this shape issue is a footgun. need to be consistent and obvious about
        # whether we are doing (X, Y, Z) or (Z, Y, X) for image as well as img_size,
        # spacing.
        data = data.T.contiguous()
        data = pad_to_shape(data, self.img_size)

        if brain_mask is None:
            mask = data > data.mean()
        else:
            brain_mask = nib.as_closest_canonical(brain_mask)
            mask_data = torch.from_numpy(brain_mask.get_fdata(dtype=np.float32))
            mask_spacing = brain_mask.header.get_zooms()
            mask_data = rescale(mask_data, mask_spacing, target_spacing=self.spacing)
            mask = mask_data.T.contiguous() > 0.5
            mask = pad_to_shape(mask.float(), self.img_size) > 0.5

        # TODO(mihir) normalization?

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

    backbone = SmriMaeBackbone(model.encoder)
    transform = SmriMaeTransform(img_size=args["img_size"])

    return transform, backbone
