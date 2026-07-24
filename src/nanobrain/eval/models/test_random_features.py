import nibabel as nib
import numpy as np
import torch
from datasets import Dataset, Features, Nifti

from nanobrain.eval.models import create_model


def test_random_features_transform_handles_non_ras():
    # HF decodes niftis to a wrapper that breaks nibabel reorientation; a non-RAS volume
    # must still load. Regression test for the DLBS decode crash.
    affine = np.diag([-1.0, -1.0, 1.0, 1.0])  # axcodes L,P,S -> forces reorientation
    data = np.random.default_rng(0).random((20, 24, 22), dtype=np.float32)
    img_bytes = nib.Nifti1Image(data, affine).to_bytes()
    dataset = Dataset.from_dict(
        {"image": [{"path": None, "bytes": img_bytes}]}, features=Features({"image": Nifti()})
    )
    wrapped = dataset[0]["image"]  # datasets Nifti1ImageWrapper, as the loader would yield

    _, transform = create_model("random_features", size=32, patch=8, dim=16)
    sample = transform(wrapped)
    assert sample["image"].shape == (1, 32, 32, 32)
    assert torch.isfinite(sample["image"]).all()


def test_random_features_contract():
    model, transform = create_model("random_features", size=32, patch=8, dim=64)
    img = nib.Nifti1Image(
        np.random.default_rng(0).random((20, 24, 22), dtype=np.float32), np.eye(4)
    )
    seg_data = np.zeros((20, 24, 22), dtype=np.float32)
    seg_data[8:12, 10:14, 9:13] = 1.0
    seg = nib.Nifti1Image(seg_data, np.eye(4))

    sample = transform(img, seg)
    batch = {"image": sample["image"].unsqueeze(0)}
    assert model.global_embed(batch).shape == (1, 64)

    patches = model.patch_embed(batch)
    labels = model.patchify_labels(sample["seg"])
    n_patches = (32 // 8) ** 3
    assert patches.shape == (1, n_patches, 64)
    assert labels.shape == (n_patches,)  # aligned with patch_embed by construction


def test_random_features_patchify_matches_grid():
    # A localized mask must produce some foreground and some background patches.
    model, transform = create_model("random_features", size=32, patch=8, dim=16)
    img = nib.Nifti1Image(np.ones((20, 24, 22), dtype=np.float32), np.eye(4))
    seg_data = np.zeros((20, 24, 22), dtype=np.float32)
    seg_data[8:12, 10:14, 9:13] = 1.0
    sample = transform(img, nib.Nifti1Image(seg_data, np.eye(4)))
    fractions = model.patchify_labels(sample["seg"])
    assert fractions.shape == ((32 // 8) ** 3,)
    assert (fractions > 0).any() and (fractions == 0).any()
