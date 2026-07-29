"""The RAS-canonical grid the seg probe aligns to, plus the preprocessing every model shares."""

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


def canonical_img(img: nib.Nifti1Image) -> nib.Nifti1Image:
    """The RAS-canonical image, affine included, for models that preprocess in world space.

    Rebuilds a plain image first because HF's Nifti1ImageWrapper reorients incorrectly.
    """
    plain = nib.Nifti1Image(img.dataobj, img.affine, img.header)
    return nib.as_closest_canonical(plain)


def canonical(img: nib.Nifti1Image) -> Tensor:
    """(X, Y, Z) float32 volume on the RAS-canonical grid.

    The axis flips leave negative strides that torch rejects, hence ascontiguousarray.
    Reorientation is a pure axis permutation, so integer label maps survive it exactly
    (round to int at the call site).
    """
    data = canonical_img(img).get_fdata(dtype=np.float32)
    return torch.from_numpy(np.ascontiguousarray(data))


def brain_mask(image: Tensor) -> Tensor:
    """Mean-threshold foreground mask, matching the intensity normalization the models use."""
    return image > image.mean()


def normalize(image: Tensor) -> Tensor:
    """Brain-masked z-score: standardize within a mean-threshold mask, zero the background."""
    brain = brain_mask(image)
    mean = image[brain].mean()
    std = image[brain].std().clamp_min(1e-6)
    return torch.where(brain, (image - mean) / std, 0.0)


def resize(volume: Tensor, size: int) -> Tensor:
    """Trilinear resample of an (X, Y, Z) volume onto a `size` cube."""
    resized = F.interpolate(
        volume[None, None], size=(size, size, size), mode="trilinear", align_corners=False
    )
    return resized[0, 0]
