from typing import Literal

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from nanobrain.eval.models import register_model
from nanobrain.eval.nifti import canonical_img

import smri_mae.model_mae as models_mae
from smri_mae.utils import filter_kwargs

PAD_TO_MULTIPLE = 32


class SmriMae(nn.Module):
    def __init__(
        self,
        encoder: models_mae.MaskedEncoder,
        transform: "SmriMaeTransform",
        global_pool: Literal["cls", "reg", "patch"] = "patch",
        pad_to_multiple: int | None = PAD_TO_MULTIPLE,
    ):
        super().__init__()
        self.encoder = encoder
        self.transform = transform
        self.global_pool = global_pool
        self.pad_to_multiple = pad_to_multiple

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @torch.inference_mode()
    def global_embed(self, img: nib.Nifti1Image) -> Tensor:
        sample = self.transform(img)
        images = sample["image"][None].to(self.device)
        mask = sample["mask"][None].to(self.device)

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
        return embed[0]  # the contract is one (D,) vector per volume

    def dense_embed(self, img: nib.Nifti1Image) -> Tensor:
        raise NotImplementedError("SmriMae does not yet implement dense embedding")


class SmriMaeTransform:
    def __init__(
        self,
        img_size: tuple[int, int, int],
        spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
        transpose: bool = False,
    ):
        self.img_size = img_size
        self.spacing = spacing
        self.transpose = transpose

    def __call__(self, img: nib.Nifti1Image) -> dict[str, Tensor]:
        # reorient to RAS
        img = canonical_img(img)

        data = img.get_fdata(dtype=np.float32)
        data = torch.from_numpy(np.ascontiguousarray(data))
        spacing = img.header.get_zooms()

        # resize
        if max(abs(s - s_) for s, s_ in zip(spacing, self.spacing)) > 0.05:
            data = rescale(data, spacing, target_spacing=self.spacing)

        # tranpose (X, Y, Z) F-order -> (Z, Y, X) C-order
        # TODO: this shape issue is a footgun. need to be consistent and obvious about
        # whether we are doing (X, Y, Z) or (Z, Y, X) for image as well as img_size,
        # spacing.
        if self.transpose:
            data = data.permute(2, 1, 0).contiguous()
        data = pad_to_shape(data, self.img_size)

        # cheap mask
        # if we want a better mask, we have to compute it here.
        # model contract is nifti image -> embedding
        mask = data > data.mean()

        # z-score over brain-mask voxels (matches pretraining); background -> 0.
        brain = data[mask]
        # population std (÷N, correction=0) to match the pretraining normalization.
        mean, std = brain.mean(), brain.std(correction=0).clamp_min(1e-6)
        data = torch.where(mask, (data - mean) / std, 0.0)

        data = data.unsqueeze(0)
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


def resolve_ckpt(ckpt_path: str) -> str:
    """A local path for a checkpoint, downloading it if it is an hf://<org>/<repo>/<file> URI."""
    from huggingface_hub import hf_hub_download

    if ckpt_path.startswith("hf://"):
        org, repo, *rest = ckpt_path.removeprefix("hf://").split("/")
        return hf_hub_download(f"{org}/{repo}", "/".join(rest))

    return ckpt_path


@register_model
def smri_mae(ckpt_path: str, global_pool: str = "patch", transpose: bool = False):
    # mmap so only the weights are read, not the optimizer state sharing the checkpoint
    path = resolve_ckpt(ckpt_path)
    ckpt = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    args = ckpt["args"]

    model_fn = models_mae.__dict__[args["model"]]
    model: models_mae.MaskedAutoencoderViT = model_fn(
        img_size=args["img_size"],
        in_chans=args.get("in_chans", 1),
        patch_size=args["patch_size"],
        # drop model kwargs the current model_mae no longer takes, so older checkpoints
        # outlive the training flags they were written with
        **filter_kwargs(models_mae.MaskedAutoencoderViT, args.get("model_kwargs") or {}),
    )
    model.load_state_dict(ckpt["model"])

    transform = SmriMaeTransform(img_size=args["img_size"], transpose=transpose)
    return SmriMae(
        model.encoder,
        transform,
        global_pool=global_pool,
        pad_to_multiple=args.get("pad_to_multiple", PAD_TO_MULTIPLE),
    )
