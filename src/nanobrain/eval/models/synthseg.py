"""SynthSeg (Billot et al., Med Image Anal 2023): a supervised 3D U-Net over synthetic scans.

The 384-channel bottleneck of the down arm is mean-pooled over the scan into one vector.
`resample` picks the interpolation backend for the 1mm resample: "verbatim" is SynthSeg's scipy
path, "torch" samples the same coordinates on the module's device.
"""

import os

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from nanobrain.eval.models import register_model

TARGET_RES = 1.0
N_LEVELS = 5
MIN_SIZE = 128
STRIDE = 2 ** (N_LEVELS - 1)  # input voxels per bottleneck cell


class SynthSeg(nn.Module):
    def __init__(self, resample: str = "torch"):
        super().__init__()
        from SynthSeg_pytorch.labels import load_synthseg_labels
        from SynthSeg_pytorch.model import build_synthseg_unet
        from SynthSeg_pytorch.predict import get_model_dir
        from SynthSeg_pytorch.weights import load_unet_from_h5

        if resample not in ("verbatim", "torch"):
            raise ValueError(f"resample should be 'verbatim' or 'torch', had {resample!r}")

        labels, _flip_indices, _topology_classes = load_synthseg_labels(fast=False)
        self.net = build_synthseg_unet(nb_labels=len(labels), name="unet")
        load_unet_from_h5(os.path.join(get_model_dir(), "synthseg_2.0.h5"), self.net)

        self.resample = resample
        self.requires_grad_(False)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @torch.inference_mode()
    def global_embed(self, img: nib.Nifti1Image) -> Tensor:
        volume, pad_idx = preprocess(img, self.device, self.resample)
        embedding, _skips = self.net.encode(volume[None, None])  # (1, 384, X/16, Y/16, Z/16)
        return embedding[0][(slice(None), *bottleneck_box(pad_idx))].mean((1, 2, 3))  # (384,)

    def dense_embed(self, img: nib.Nifti1Image) -> Tensor:
        raise NotImplementedError(
            "SynthSeg predicts on its own 1mm grid after resampling and padding, and postprocess "
            "never resamples back, so mapping the decoder onto the canonical grid is unbuilt."
        )


def preprocess(
    img: nib.Nifti1Image, device: torch.device, resample: str
) -> tuple[Tensor, np.ndarray]:
    """`SynthSeg.predict_synthseg.preprocess` for an in-memory nifti, on their default path.

    Returns the (X, Y, Z) network input on `device`, and the indices the scan occupies within it.
    """
    from SynthSeg_pytorch.preprocessing import (
        align_volume_to_ref,
        find_closest_number_divisible_by_m,
        get_volume_info,
        pad_volume,
        rescale_volume,
        resample_volume,
    )

    volume, affine, n_dims, _n_channels, _header, res = get_volume_info(img)
    if n_dims != 3:
        raise ValueError(f"input should have 3 dimensions, had {n_dims}")

    if np.any(np.abs(res - TARGET_RES) > 0.05):
        if resample == "torch":
            volume, affine = resample_torch(volume, affine, device)
        else:
            volume, affine = resample_volume(volume, affine, [TARGET_RES] * n_dims)

    volume = align_volume_to_ref(volume, affine, np.eye(4), n_dims=n_dims, return_copy=False)
    volume = rescale_volume(
        volume, new_min=0.0, new_max=1.0, min_percentile=0.5, max_percentile=99.5
    )

    pad_shape = [find_closest_number_divisible_by_m(s, 2**N_LEVELS, "higher") for s in volume.shape]
    volume, pad_idx = pad_volume(volume, np.maximum(pad_shape, MIN_SIZE), return_pad_idx=True)
    return torch.as_tensor(volume, dtype=torch.float32, device=device), pad_idx


def bottleneck_box(pad_idx: np.ndarray) -> tuple[slice, ...]:
    """The bottleneck cells whose input voxels lie inside the scan, so pooling skips the padding.

    Approximate either way: a cell's receptive field reaches beyond the STRIDE voxels it covers.
    """
    box = []
    for low, high in zip(pad_idx[:3], pad_idx[3:]):
        start = -(-int(low) // STRIDE)
        box.append(slice(start, max(int(high) // STRIDE, start + 1)))
    return tuple(box)


def resample_torch(
    volume: np.ndarray, affine: np.ndarray, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """`resample_volume` with the interpolation on `device`.

    `align_corners=False` samples the coordinates the reference samples, over `floor(n * factor)`
    of them rather than `ceil`. The reference edge-clamps the coordinates it samples beyond the
    last voxel, so the source is replicated far enough for `F.interpolate` to read those values
    where it expects them, and the extra output trimmed back to the reference's length.
    """
    pixdim = np.sqrt(np.sum(affine * affine, axis=0))[:-1]
    factor = pixdim / TARGET_RES
    sigmas = 0.25 / factor
    sigmas[factor > 1] = 0  # don't blur if upsampling

    padding = [0] * 6
    for axis, scale in enumerate(factor):
        padding[2 * (2 - axis) + 1] = int(np.ceil(1.0 / scale))

    resampled = torch.as_tensor(volume, dtype=torch.float32, device=device)[None, None]
    resampled = gaussian_blur(resampled, sigmas)
    resampled = F.interpolate(
        F.pad(resampled, padding, mode="replicate"),
        scale_factor=tuple(factor),
        mode="trilinear",
        align_corners=False,
        recompute_scale_factor=False,
    )
    size = np.ceil(np.array(volume.shape) * factor).astype(int)
    resampled = resampled[:, :, : size[0], : size[1], : size[2]]

    affine = affine.copy()
    affine[:-1, :-1] = affine[:-1, :-1] / factor
    affine[:-1, -1] = affine[:-1, -1] - affine[:-1, :-1] @ (0.5 * (factor - 1))
    return resampled[0, 0].cpu().numpy().astype(np.float64), affine


def gaussian_blur(volume: Tensor, sigmas: np.ndarray) -> Tensor:
    """Separable gaussian blur of a (1, 1, X, Y, Z) volume, one axis per non-zero sigma."""
    for axis, sigma in enumerate(sigmas):
        if sigma <= 0:
            continue
        radius = int(4.0 * sigma + 0.5)  # scipy's default truncation
        offsets = torch.arange(-radius, radius + 1, dtype=volume.dtype, device=volume.device)
        kernel = torch.exp(-0.5 * (offsets / sigma) ** 2)
        kernel = kernel / kernel.sum()

        shape = [1, 1, 1, 1, 1]
        shape[2 + axis] = -1
        padding = [0, 0, 0, 0, 0, 0]
        padding[2 * (2 - axis)] = padding[2 * (2 - axis) + 1] = radius
        volume = F.conv3d(F.pad(volume, padding, mode="replicate"), kernel.reshape(shape))
    return volume


@register_model
def synthseg(resample: str = "torch") -> SynthSeg:
    return SynthSeg(resample=resample)
