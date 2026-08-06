import itertools

import nibabel as nib
import numpy as np
import torch
from datasets import Dataset, Features, Nifti
from nibabel.affines import apply_affine

from nanobrain.eval.models import create_model
from nanobrain.eval.models.base import PatchFeatures
from nanobrain.eval.nifti import canonical_img


def world_bounds(img: nib.Nifti1Image) -> tuple[np.ndarray, np.ndarray]:
    """World-mm bounding box of the canonical grid's voxel centres."""
    canon = canonical_img(img)
    corners = apply_affine(
        canon.affine, np.array(list(itertools.product(*[(0, s - 1) for s in canon.shape])))
    )
    return corners.min(axis=0), corners.max(axis=0)


def assert_patches_span_the_image(patches: PatchFeatures, img: nib.Nifti1Image, patch_mm: float):
    """Patch centres land on the image, within one patch of its bounding box."""
    features, coords = patches
    assert len(features) == len(coords)
    assert coords.shape[1] == 3
    assert torch.isfinite(coords).all()
    low, high = world_bounds(img)
    assert (coords.numpy() >= low - patch_mm).all() and (coords.numpy() <= high + patch_mm).all()


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
    patches = model.patch_embed(wrapped)
    assert patches.features.shape == (3 * 3 * 3, 16)  # ceil(20/8) x ceil(24/8) x ceil(22/8)
    assert torch.isfinite(patches.features).all()
    assert_patches_span_the_image(patches, wrapped, patch_mm=8)


def test_random_features_contract():
    model = create_model("random_features", size=32, patch=8, dim=64, patch_dim=16)
    img = nib.Nifti1Image(
        np.random.default_rng(0).random((20, 24, 22), dtype=np.float32), np.eye(4)
    )
    assert model.global_embed(img).shape == (64,)

    patches = model.patch_embed(img)
    assert patches.features.shape == (3 * 3 * 3, 16)  # one patch_dim vector per 8^3 block
    assert patches.features.device.type == "cpu" and patches.coords.device.type == "cpu"
    assert torch.isfinite(patches.features).all()
    assert patches.features.std() > 0  # features vary over the volume, not a collapsed constant
    assert_patches_span_the_image(patches, img, patch_mm=8)


def test_random_features_patch_grid_not_multiple_of_patch():
    # A grid not divisible by patch is padded up, so the last block along each axis still exists.
    model = create_model("random_features", size=32, patch=8, dim=16, patch_dim=16)
    data = np.random.default_rng(0).random((13, 17, 11), dtype=np.float32)
    img = nib.Nifti1Image(data, np.eye(4))
    patches = model.patch_embed(img)
    assert patches.features.shape == (2 * 3 * 2, 16)
    assert_patches_span_the_image(patches, img, patch_mm=8)


def test_random_features_coords_follow_the_affine():
    # Patch centres are world mm, so a spacing/offset change must move them by exactly that much.
    model = create_model("random_features", size=32, patch=8, dim=16, patch_dim=16)
    data = np.random.default_rng(0).random((16, 16, 16), dtype=np.float32)
    affine = np.diag([2.0, 3.0, 1.5, 1.0])
    affine[:3, 3] = [-30.0, -40.0, -20.0]

    identity = model.patch_embed(nib.Nifti1Image(data, np.eye(4))).coords.numpy()
    scaled = model.patch_embed(nib.Nifti1Image(data, affine)).coords.numpy()
    expected = identity * np.array([2.0, 3.0, 1.5]) + np.array([-30.0, -40.0, -20.0])
    assert np.allclose(scaled, expected, atol=1e-4)
