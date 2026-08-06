"""Neuro-JEPA (NYUMedML): a JEPA-pretrained 3D ViT-B/12 with a sparse MoE, frozen.

Weights are gated on the Hub, so `HF_TOKEN` must be set and access granted. The 576 tokens
(an 8x9x8 grid of 12-voxel patches over a 96x108x96 crop) are mean-pooled into one vector for
`global_embed`, or returned with their world-mm centres for `patch_embed`. A token spans ~21mm of
the original scan, so expect coarse segmentations.
The model expects 1mm scans affine-registered to MNI152, which is what it was pretrained on.
"""

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from nibabel.affines import apply_affine
from torch import Tensor

from nanobrain.eval.models import register_model
from nanobrain.eval.models.base import PatchFeatures
from nanobrain.eval.nifti import canonical_img

REPO_ID = "NYUMedML/Neuro-JEPA"
IMG_SIZE = (96, 108, 96)


class NeuroJEPA(nn.Module):
    def __init__(self):
        super().__init__()
        from neurojepa.utils.init_utils import load_backbone_from_hf

        self.backbone = load_backbone_from_hf(REPO_ID, device="cpu")
        self.backbone.out_layers = None  # pool the last block, whatever the hub config says
        self.transform = static_transform()
        self.requires_grad_(False)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @torch.inference_mode()
    def global_embed(self, img: nib.Nifti1Image) -> Tensor:
        volume, _affine = preprocess(self.transform, img, self.device)  # (1, 1, 96, 108, 96)
        tokens, _moe_scores = self.backbone(volume)
        return tokens[0].mean(0)  # (768,)

    @torch.inference_mode()
    def patch_embed(self, img: nib.Nifti1Image) -> PatchFeatures:
        volume, affine = preprocess(self.transform, img, self.device)
        tokens, _moe_scores = self.backbone(volume)  # (1, 576, 768)

        # PatchEmbed3D flattens the patch grid in C order, so tokens follow "x y z ->" as well
        patch = np.array(self.backbone.patch_embed.patch_size)
        grid = np.array(volume.shape[2:]) // patch
        assert grid.prod() == tokens.shape[1], f"{tokens.shape[1]} tokens over a {grid} patch grid"
        centres = rearrange(np.indices(tuple(grid)), "c x y z -> (x y z) c") * patch
        coords = apply_affine(affine, centres + (patch - 1) / 2)
        return PatchFeatures(tokens[0].float().cpu(), torch.from_numpy(coords).float())


def preprocess(transform, img: nib.Nifti1Image, device: torch.device) -> tuple[Tensor, np.ndarray]:
    """Neuro-JEPA's static pipeline: a nifti to a (1, 1, X, Y, Z) batch of one, and the affine
    mapping that volume's voxels to RAS world mm.

    The affine rides along on a MetaTensor: without it the resample to 1mm silently no-ops, and
    it is also what places the tokens in world space.
    Runs on `device` because the bspline resample dominates: 7.0s on CPU, 0.19s on GPU.
    """
    from monai.data import MetaTensor

    canon = canonical_img(img)
    data = np.nan_to_num(canon.get_fdata(dtype=np.float32))
    volume = torch.from_numpy(np.ascontiguousarray(data))[None].to(device)  # (1, X, Y, Z)
    volume = MetaTensor(volume, affine=torch.as_tensor(canon.affine).to(device))
    out = transform(volume)
    return out.as_tensor()[None], np.asarray(out.affine.cpu())


def static_transform():
    """Mirrors `loading_transforms` + test-mode `vit3d_transforms` in the neurojepa package.

    Those are dict transforms over a path, so they cannot take the in-memory nifti the eval
    harness hands us; `test_neurojepa.py` checks this chain against them voxel for voxel.
    """
    from monai import transforms

    return transforms.Compose(
        [
            transforms.ScaleIntensityRangePercentiles(
                lower=0.5, upper=99.5, b_min=0, b_max=1, clip=True
            ),
            transforms.ResizeWithPadOrCrop(spatial_size=(180, 216, 180), mode="edge"),
            transforms.Spacing(pixdim=(1.0, 1.0, 1.0), mode=5),
            transforms.CropForeground(select_fn=lambda x: x > 0.0, margin=4, allow_smaller=True),
            transforms.Resize(spatial_size=(100, 120, 100)),
            transforms.CastToType(dtype=np.float32),
            transforms.ResizeWithPadOrCrop(spatial_size=IMG_SIZE, mode="constant", value=0),
            transforms.CenterSpatialCrop(roi_size=IMG_SIZE),
        ]
    )


@register_model
def neurojepa() -> NeuroJEPA:
    return NeuroJEPA()
