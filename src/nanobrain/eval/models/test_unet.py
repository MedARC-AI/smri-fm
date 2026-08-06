import nibabel as nib
import numpy as np
import torch
from datasets import Dataset, Features, Nifti

from nanobrain.eval.models import create_model

SMALL = {"size": 16, "base": 4, "levels": 3, "pool": 3}
GLOBAL_DIM = 16  # the deepest stage's width, one centre cell


def make_image(shape: tuple[int, int, int], affine: np.ndarray = np.eye(4)) -> nib.Nifti1Image:
    data = np.random.default_rng(0).random(shape, dtype=np.float32)
    return nib.Nifti1Image(data, affine)


def test_unet_handles_non_ras():
    # HF decodes niftis to a wrapper that breaks nibabel reorientation; a non-RAS volume
    # must still load. Regression test for the DLBS decode crash.
    affine = np.diag([-1.0, -1.0, 1.0, 1.0])  # axcodes L,P,S -> forces reorientation
    img_bytes = make_image((20, 24, 22), affine).to_bytes()
    dataset = Dataset.from_dict(
        {"image": [{"path": None, "bytes": img_bytes}]}, features=Features({"image": Nifti()})
    )
    wrapped = dataset[0]["image"]  # datasets Nifti1ImageWrapper, as the loader would yield

    model = create_model("random_unet", **SMALL)
    assert model.global_embed(wrapped).shape == (GLOBAL_DIM,)
    dense = model.dense_embed(wrapped)
    assert dense.shape == (20, 24, 22, 4)  # canonicalized grid (L,P,S flips keep the shape)
    assert torch.isfinite(dense).all()


def test_unet_contract():
    model = create_model("random_unet", **SMALL)
    img = make_image((20, 24, 22))
    assert model.global_embed(img).shape == (GLOBAL_DIM,)

    dense = model.dense_embed(img)
    assert dense.shape == (20, 24, 22, 4)  # one base-width vector per voxel on the input grid
    assert dense.device.type == "cpu"
    assert torch.isfinite(dense).all()
    assert dense.std() > 0  # features vary over the volume, not a collapsed constant map


def test_unet_dense_grid_not_multiple_of_stride():
    # Odd sizes survive the stride-2 encoder because the decoder upsamples to each skip's grid.
    model = create_model("random_unet", **SMALL)
    assert model.dense_embed(make_image((13, 17, 11))).shape == (13, 17, 11, 4)


def test_unet_param_budget():
    model = create_model("random_unet")
    assert sum(p.numel() for p in model.parameters()) < 20e6
