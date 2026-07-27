"""The RAS-canonical grid: the shared frame that dense_embed and the seg probe align to."""

import nibabel as nib
import numpy as np
import torch
from torch import Tensor


def canonical(img: nib.Nifti1Image) -> Tensor:
    """(X, Y, Z) float32 volume on the RAS-canonical grid.

    Rebuilds a plain image first because HF's Nifti1ImageWrapper reorients incorrectly; the axis
    flips leave negative strides that torch rejects, hence ascontiguousarray. Reorientation is a
    pure axis permutation, so integer label maps survive it exactly (round to int at the call site).
    """
    plain = nib.Nifti1Image(img.dataobj, img.affine, img.header)
    data = nib.as_closest_canonical(plain).get_fdata(dtype=np.float32)
    return torch.from_numpy(np.ascontiguousarray(data))


def brain_mask(image: Tensor) -> Tensor:
    """Mean-threshold foreground mask, matching the intensity normalization the models use."""
    return image > image.mean()
