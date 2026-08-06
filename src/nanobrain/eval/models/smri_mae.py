"""The wrapper around the vendored `smri_mae` package: a 3D ViT MAE encoder, frozen.

Volumes are resampled to the checkpoint's spacing on the RAS-canonical grid, padded or cropped
to its pretraining shape, and z-scored inside the brain. The unmasked patch tokens are then
mean-pooled into one vector for `global_embed`, or returned with their world-mm centres for
`patch_embed`. The brain mask is a mean threshold, not the SynthSeg mask used in pretraining, so
it keeps skull and neck.
"""

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from nibabel.affines import apply_affine
from torch import Tensor

from nanobrain.eval.models import register_model
from nanobrain.eval.models.base import PatchFeatures
from nanobrain.eval.nifti import canonical_img

import smri_mae.model_mae as models_mae
from smri_mae.utils import filter_kwargs


class SmriMae(nn.Module):
    def __init__(self, encoder: models_mae.MaskedEncoder, transform: "SmriMaeTransform"):
        super().__init__()
        self.encoder = encoder
        self.transform = transform
        self.requires_grad_(False)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @torch.inference_mode()
    def global_embed(self, img: nib.Nifti1Image) -> Tensor:
        sample = self.transform(img)
        images = sample["image"][None].to(self.device)
        mask = sample["mask"][None].to(self.device)

        _cls, _reg, patch, _, _, token_mask = self.encoder(images, mask=mask)
        token_mask = token_mask.to(device=patch.device, dtype=torch.bool)
        denom = token_mask.sum(dim=1, keepdim=True).clamp(min=1).to(dtype=patch.dtype)
        embed = (patch * token_mask.unsqueeze(-1)).sum(dim=1) / denom
        return embed[0]  # the contract is one (D,) vector per volume

    @torch.inference_mode()
    def patch_embed(self, img: nib.Nifti1Image) -> PatchFeatures:
        """The brain-holding patch tokens, which are the ones `global_embed` pools and the ones
        pretraining ever saw. `patch_ids` is what pairs them back to the grid."""
        sample = self.transform(img)
        images = sample["image"][None].to(self.device)
        mask = sample["mask"][None].to(self.device)

        _cls, _reg, patch, _, patch_ids, token_mask = self.encoder(images, mask=mask)
        token_mask = token_mask.to(device=patch.device, dtype=torch.bool)[0]

        patchify = self.encoder.patchify
        grid, size = np.array(patchify.grid_size), np.array(patchify.patch_size)
        assert tuple(grid * size) == tuple(images.shape[2:]), "token grid must tile the volume"
        # `patchify3d` flattens the grid as "(t h w)", so the centres follow "x y z ->" as well
        centres = rearrange(np.indices(tuple(grid)), "c x y z -> (x y z) c") * size
        coords = apply_affine(sample["affine"].numpy(), centres + (size - 1) / 2)

        kept = patch_ids[0][token_mask].cpu().numpy()
        return PatchFeatures(
            patch[0][token_mask].float().cpu(), torch.from_numpy(coords[kept]).float()
        )


class SmriMaeTransform:
    def __init__(
        self,
        img_size: tuple[int, int, int],
        spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ):
        self.img_size = img_size
        self.spacing = spacing

    def __call__(self, img: nib.Nifti1Image) -> dict[str, Tensor]:
        """The fitted, z-scored volume, its brain mask, and the affine placing that grid's voxels
        in RAS world mm."""
        img = canonical_img(img)
        data = torch.from_numpy(np.ascontiguousarray(img.get_fdata(dtype=np.float32)))
        affine = np.asarray(img.affine, dtype=float)

        spacing = img.header.get_zooms()
        if max(abs(s - s_) for s, s_ in zip(spacing, self.spacing)) > 0.05:
            data, affine = rescale(data, affine, spacing, target_spacing=self.spacing)
        data, affine = fit_to_shape(data, affine, self.img_size)

        mask = data > data.mean()
        brain = data[mask]
        # population std (correction=0) to match the pretraining normalization
        mean, std = brain.mean(), brain.std(correction=0).clamp_min(1e-6)
        data = torch.where(mask, (data - mean) / std, 0.0)
        return {
            "image": data.unsqueeze(0),
            "mask": mask.unsqueeze(0),
            "affine": torch.from_numpy(affine),
        }


def rescale(
    x: torch.Tensor,
    affine: np.ndarray,
    spacing: tuple[float, ...],
    target_spacing: tuple[float, ...] = (1.0, 1.0, 1.0),
) -> tuple[torch.Tensor, np.ndarray]:
    scales = tuple([current / target for current, target in zip(spacing, target_spacing)])
    resampled = F.interpolate(x[None, None], scale_factor=scales, mode="trilinear").squeeze(0, 1)

    # align_corners=False reads output voxel j from input voxel (j + 0.5) / scale - 0.5
    scale = np.asarray(scales, dtype=float)
    step = np.diag([*(1 / scale), 1.0])
    step[:3, 3] = 0.5 / scale - 0.5
    return resampled, affine @ step


def fit_to_shape(
    x: torch.Tensor, affine: np.ndarray, target_shape: tuple[int, ...]
) -> tuple[torch.Tensor, np.ndarray]:
    """Centre the volume in `target_shape`, padding the short axes and cropping the long ones."""
    pads = [target - size for size, target in zip(x.shape, target_shape)]
    padding = [side for pad in reversed(pads) for side in (pad // 2, pad - pad // 2)]

    # a crop is a negative pad, so output voxel k came from input voxel k - pad // 2 either way
    step = np.eye(4)
    step[:3, 3] = [-(pad // 2) for pad in pads]
    return F.pad(x, padding), affine @ step


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
    return SmriMae(model.encoder, SmriMaeTransform(img_size=args["img_size"]))
