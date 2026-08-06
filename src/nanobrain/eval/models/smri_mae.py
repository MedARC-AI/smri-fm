"""The wrapper around the vendored `smri_mae` package: a 3D ViT MAE encoder, frozen.

Volumes are resampled to the checkpoint's spacing on the RAS-canonical grid, padded or cropped
to its pretraining shape, and z-scored inside the brain. The unmasked patch tokens are then
mean-pooled into one vector. The brain mask is a mean threshold, not the SynthSeg mask used in
pretraining -- see `.claude/memory/smri-mae-preprocessing-gap.md`.
"""

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
        pad_to_multiple: int | None = PAD_TO_MULTIPLE,
    ):
        super().__init__()
        self.encoder = encoder
        self.transform = transform
        self.pad_to_multiple = pad_to_multiple
        self.requires_grad_(False)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @torch.inference_mode()
    def global_embed(self, img: nib.Nifti1Image) -> Tensor:
        sample = self.transform(img)
        images = sample["image"][None].to(self.device)
        mask = sample["mask"][None].to(self.device)

        _cls, _reg, patch, _, _, token_mask = self.encoder(
            images, mask=mask, pad_to_multiple=self.pad_to_multiple
        )
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
    ):
        self.img_size = img_size
        self.spacing = spacing

    def __call__(self, img: nib.Nifti1Image) -> dict[str, Tensor]:
        img = canonical_img(img)
        data = torch.from_numpy(np.ascontiguousarray(img.get_fdata(dtype=np.float32)))

        spacing = img.header.get_zooms()
        if max(abs(s - s_) for s, s_ in zip(spacing, self.spacing)) > 0.05:
            data = rescale(data, spacing, target_spacing=self.spacing)
        data = fit_to_shape(data, self.img_size)

        mask = data > data.mean()
        brain = data[mask]
        # population std (correction=0) to match the pretraining normalization
        mean, std = brain.mean(), brain.std(correction=0).clamp_min(1e-6)
        data = torch.where(mask, (data - mean) / std, 0.0)
        return {"image": data.unsqueeze(0), "mask": mask.unsqueeze(0)}


def rescale(
    x: torch.Tensor,
    spacing: tuple[float, ...],
    target_spacing: tuple[float, ...] = (1.0, 1.0, 1.0),
) -> torch.Tensor:
    scales = tuple([current / target for current, target in zip(spacing, target_spacing)])
    return F.interpolate(x[None, None], scale_factor=scales, mode="trilinear").squeeze(0, 1)


def fit_to_shape(x: torch.Tensor, target_shape: tuple[int, ...]) -> torch.Tensor:
    """Centre the volume in `target_shape`, padding the short axes and cropping the long ones."""
    padding = []
    for size, target in reversed(list(zip(x.shape, target_shape))):
        pad = target - size
        padding.extend([pad // 2, pad - pad // 2])
    return F.pad(x, padding)


def resolve_ckpt(ckpt_path: str) -> str:
    """A local path for a checkpoint, downloading it if it is an hf://<org>/<repo>/<file> URI."""
    from huggingface_hub import hf_hub_download

    if ckpt_path.startswith("hf://"):
        org, repo, *rest = ckpt_path.removeprefix("hf://").split("/")
        return hf_hub_download(f"{org}/{repo}", "/".join(rest))

    return ckpt_path


@register_model
def smri_mae(ckpt_path: str) -> SmriMae:
    path = resolve_ckpt(ckpt_path)
    ckpt = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    args = ckpt["args"]

    model_fn = models_mae.__dict__[args["model"]]
    model: models_mae.MaskedAutoencoderViT = model_fn(
        img_size=args["img_size"],
        in_chans=args.get("in_chans", 1),
        patch_size=args["patch_size"],
        # older checkpoints carry training flags the current model_mae no longer takes
        **filter_kwargs(models_mae.MaskedAutoencoderViT, args.get("model_kwargs") or {}),
    )
    model.load_state_dict(ckpt["model"])
    return SmriMae(
        model.encoder,
        SmriMaeTransform(img_size=args["img_size"]),
        pad_to_multiple=args.get("pad_to_multiple", PAD_TO_MULTIPLE),
    )
