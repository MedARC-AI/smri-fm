import nibabel as nib
import numpy as np
import torch
from datasets import Dataset, Features, Nifti

from nanobrain.eval.models import create_model


def test_random_features_model_handles_non_ras():
    # HF decodes niftis to a wrapper that breaks nibabel reorientation; a non-RAS volume
    # must still load. Regression test for the DLBS decode crash.
    affine = np.diag([-1.0, -1.0, 1.0, 1.0])  # axcodes L,P,S -> forces reorientation
    data = np.random.default_rng(0).random((20, 24, 22), dtype=np.float32)
    img_bytes = nib.Nifti1Image(data, affine).to_bytes()
    dataset = Dataset.from_dict(
        {"image": [{"path": None, "bytes": img_bytes}]}, features=Features({"image": Nifti()})
    )
    wrapped = dataset[0]["image"]  # datasets Nifti1ImageWrapper, as the loader would yield

    model = create_model("random_features", size=32, patch=8, dim=16, patch_dim=16)
    assert model.global_embed(wrapped).shape == (16,)
    dense = model.dense_embed(wrapped)
    assert dense.shape == (20, 24, 22, 16)  # canonicalized grid (L,P,S flips keep the shape)
    assert torch.isfinite(dense).all()


def test_random_features_contract():
    model = create_model("random_features", size=32, patch=8, dim=64, patch_dim=16)
    img = nib.Nifti1Image(
        np.random.default_rng(0).random((20, 24, 22), dtype=np.float32), np.eye(4)
    )
    assert model.global_embed(img).shape == (64,)

    dense = model.dense_embed(img)
    assert dense.shape == (20, 24, 22, 16)  # one patch_dim vector per voxel on the input grid
    assert dense.device.type == "cpu"
    assert torch.isfinite(dense).all()


def test_random_features_dense_grid_not_multiple_of_patch():
    # A grid not divisible by patch must still yield per-voxel features on the input shape.
    model = create_model("random_features", size=32, patch=8, dim=16, patch_dim=16)
    data = np.random.default_rng(0).random((13, 17, 11), dtype=np.float32)
    assert model.dense_embed(nib.Nifti1Image(data, np.eye(4))).shape == (13, 17, 11, 16)
