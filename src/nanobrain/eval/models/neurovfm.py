"""NeuroVFM (MLNeurosurg): a ViT-B over 4x16x16 patches of a 1x1x4mm volume, frozen.

Tokens are packed varlen with their 3D coordinates and background tokens are dropped, so the
count varies per scan: they are mean-pooled into one vector for `global_embed`, or returned with
their world-mm centres for `patch_embed`. A token covers a 16mm cube of the scan, and dropping
the background leaves a brain-shaped point cloud rather than a full grid.
Weights are public and ungated.
"""

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from nibabel.affines import apply_affine
from torch import Tensor

from nanobrain.eval.models import register_model
from nanobrain.eval.models.base import PatchFeatures

REPO_ID = "mlinslab/neurovfm-encoder"

# `transpose_to_dhw` rolls the 4mm axis to the front of SimpleITK's (z, y, x) array, so the
# resulting [D, H, W] axes are these SimpleITK (x, y, z) index axes, one entry per `view`.
SITK_AXES = {0: [2, 1, 0], 1: [1, 2, 0], 2: [0, 1, 2]}


class NeuroVFM(nn.Module):
    def __init__(self):
        super().__init__()
        from neurovfm.pipelines.encoder import load_encoder

        pipeline, self.preproc = load_encoder(REPO_ID, device="cpu")
        self.backbone = pipeline.model
        self.norm_module = pipeline.norm_module
        self.requires_grad_(False)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def encode(self, batch: dict) -> Tensor:
        """(N, 768) one embedding per foreground token, in `batch["coords"]` order."""
        tokens = batch["img"].to(self.device)
        coords = batch["coords"].to(self.device)
        cu_seqlens = batch["series_cu_seqlens"].to(self.device)
        tokens = self.norm_module.normalize(
            tokens, batch["mode"], batch["path"], cu_seqlens=cu_seqlens, sizes=batch["size"]
        )
        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            embs = self.backbone(  # masks=None because background is already dropped
                tokens,
                coords,
                masks=None,
                cu_seqlens=cu_seqlens,
                max_seqlen=batch["series_max_len"],
                use_flash_attn=False,
            )
        return embs.float()

    @torch.inference_mode()
    def global_embed(self, img: nib.Nifti1Image) -> Tensor:
        return self.encode(preprocess(self.preproc, img)).mean(0)

    @torch.inference_mode()
    def patch_embed(self, img: nib.Nifti1Image) -> PatchFeatures:
        batch = preprocess(self.preproc, img)
        return PatchFeatures(self.encode(batch).cpu(), batch["world"])


def preprocess(preproc, img: nib.Nifti1Image) -> dict:
    """NeuroVFM's `StudyPreprocessor.load_study` for a single in-memory volume.

    That method globs a directory and reads with SimpleITK, so it cannot take the nifti the
    eval harness hands us; `test_neurovfm.py` checks this chain against it token for token.
    """
    from neurovfm.data.preprocess import prepare_for_inference, tokenize_volume
    from neurovfm.data.utils import preprocess_image

    img_sitk = preprocess_image(to_sitk(img))
    img_arrs, background_mask, view = prepare_for_inference(img_sitk, mode="mri")
    tokens, coords, _filtered = tokenize_volume(
        img_arrs[0],
        background_mask,
        patch_size=preproc.patch_size,
        remove_background=preproc.remove_background,
    )
    world = patch_centres_mm(img_sitk, view, coords, preproc.patch_size)
    return {
        "img": torch.from_numpy(tokens).float(),
        "coords": torch.from_numpy(coords).long(),
        "world": torch.from_numpy(world).float(),
        "series_cu_seqlens": torch.tensor([0, len(tokens)], dtype=torch.int32),
        "series_max_len": len(tokens),
        "mode": ["mri"],
        "path": ["<in-memory>"],
        "size": [img_arrs[0].shape],
    }


def patch_centres_mm(
    img_sitk, view: int, coords: np.ndarray, patch_size: tuple[int, int, int]
) -> np.ndarray:
    """(N, 3) RAS world mm centre of each token, from its (d, h, w) patch-grid index."""
    patch = np.array(patch_size)
    centres = coords * patch + (patch - 1) / 2  # voxel centre of the patch in the [D, H, W] array
    index = np.empty_like(centres)
    index[:, SITK_AXES[view]] = centres  # (d, h, w) back onto SimpleITK's (x, y, z) index order
    return apply_affine(from_sitk_affine(img_sitk), index)


def from_sitk_affine(img_sitk) -> np.ndarray:
    """A SimpleITK image's (x, y, z) index -> RAS world mm affine, inverting `to_sitk`."""
    direction = np.array(img_sitk.GetDirection()).reshape(3, 3)
    affine = np.eye(4)
    affine[:3, :3] = direction * np.array(img_sitk.GetSpacing())
    affine[:3, 3] = img_sitk.GetOrigin()
    return np.diag([-1.0, -1.0, 1.0, 1.0]) @ affine


def to_sitk(img: nib.Nifti1Image):
    """A nifti as a SimpleITK image, converting the RAS affine to SimpleITK's LPS origin/direction."""
    import SimpleITK as sitk

    lps = np.diag([-1.0, -1.0, 1.0, 1.0]) @ img.affine
    spacing = np.linalg.norm(lps[:3, :3], axis=0)
    data = np.asanyarray(img.dataobj, dtype=np.float32)
    out = sitk.GetImageFromArray(np.ascontiguousarray(data.transpose(2, 1, 0)))
    out.SetSpacing(spacing.tolist())
    out.SetDirection((lps[:3, :3] / spacing).ravel().tolist())
    out.SetOrigin(lps[:3, 3].tolist())
    return out


@register_model
def neurovfm() -> NeuroVFM:
    return NeuroVFM()
