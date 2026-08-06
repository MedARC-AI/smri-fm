"""NeuroVFM (MLNeurosurg): a ViT-B over 4x16x16 patches of a 1x1x4mm volume, frozen.

Tokens are packed varlen with their 3D coordinates and background tokens are dropped, so the
count varies per scan; they are mean-pooled into one vector. Weights are public and ungated.
"""

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from nanobrain.eval.models import register_model
from nanobrain.eval.models.base import PatchFeatures

REPO_ID = "mlinslab/neurovfm-encoder"


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

    @torch.inference_mode()
    def global_embed(self, img: nib.Nifti1Image) -> Tensor:
        batch = preprocess(self.preproc, img)
        tokens = batch["img"].to(self.device)
        coords = batch["coords"].to(self.device)
        cu_seqlens = batch["series_cu_seqlens"].to(self.device)
        tokens = self.norm_module.normalize(
            tokens, batch["mode"], batch["path"], cu_seqlens=cu_seqlens, sizes=batch["size"]
        )
        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            embs = self.backbone(  # (N, 768); masks=None because background is already dropped
                tokens,
                coords,
                masks=None,
                cu_seqlens=cu_seqlens,
                max_seqlen=batch["series_max_len"],
                use_flash_attn=False,
            )
        return embs.float().mean(0)

    def patch_embed(self, img: nib.Nifti1Image) -> PatchFeatures:
        raise NotImplementedError(
            "not yet ported to the patch contract: `preprocess` already returns per-token coords, "
            "which need mapping through the preprocessed sitk image's geometry into RAS world mm."
        )


def preprocess(preproc, img: nib.Nifti1Image) -> dict:
    """NeuroVFM's `StudyPreprocessor.load_study` for a single in-memory volume.

    That method globs a directory and reads with SimpleITK, so it cannot take the nifti the
    eval harness hands us; `test_neurovfm.py` checks this chain against it token for token.
    """
    from neurovfm.data.preprocess import prepare_for_inference, tokenize_volume
    from neurovfm.data.utils import preprocess_image

    img_sitk = preprocess_image(to_sitk(img))
    img_arrs, background_mask, _view = prepare_for_inference(img_sitk, mode="mri")
    tokens, coords, _filtered = tokenize_volume(
        img_arrs[0],
        background_mask,
        patch_size=preproc.patch_size,
        remove_background=preproc.remove_background,
    )
    return {
        "img": torch.from_numpy(tokens).float(),
        "coords": torch.from_numpy(coords).long(),
        "series_cu_seqlens": torch.tensor([0, len(tokens)], dtype=torch.int32),
        "series_max_len": len(tokens),
        "mode": ["mri"],
        "path": ["<in-memory>"],
        "size": [img_arrs[0].shape],
    }


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
