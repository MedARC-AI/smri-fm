"""The wrapper around the vendored `smri_mae` package.

The pretrained checkpoints are too big to test against, so these cover the parts we wrote -- the
transform, the pooling contract, and checkpoint path resolution -- on a randomly-initialized
depth-0 encoder. Depth 0 keeps the tokenizing, masking and pooling while skipping the blocks:
their jagged SDPA asks the cuDNN backend whether it can run, which errors out on a machine with
a CUDA-built torch but no device.
"""

import nibabel as nib
import numpy as np
import pytest
import torch
from datasets import Dataset, Features, Nifti

import smri_mae.model_mae as models_mae
from nanobrain.eval.models.smri_mae import SmriMae, SmriMaeTransform, resolve_ckpt

IMG_SIZE = (32, 40, 32)
EMBED_DIM = 32


def make_model(global_pool: str = "patch") -> SmriMae:
    mae = models_mae.MaskedAutoencoderViT(
        img_size=IMG_SIZE,
        patch_size=8,
        depth=0,
        embed_dim=EMBED_DIM,
        num_heads=2,
        decoder_depth=0,
        decoder_embed_dim=EMBED_DIM,
        decoder_num_heads=2,
    )
    transform = SmriMaeTransform(img_size=IMG_SIZE)
    return SmriMae(mae.encoder, transform, global_pool=global_pool).eval()


def make_image(shape: tuple[int, int, int], affine: np.ndarray = np.eye(4)) -> nib.Nifti1Image:
    """A bright box in a noisy background, at raw MRI intensities the transform normalizes away.

    The box is half the volume along each axis, so the mean threshold recovers it exactly and
    its extent tracks the resampling.
    """
    data = np.random.default_rng(0).random(shape, dtype=np.float32) * 50
    box = tuple(slice(size // 4, size // 4 + size // 2) for size in shape)
    data[box] += 800
    return nib.Nifti1Image(data, affine)


@pytest.mark.parametrize("global_pool", ["cls", "patch"])
def test_global_embed_contract(global_pool):
    embed = make_model(global_pool).global_embed(make_image((28, 36, 30)))
    assert embed.shape == (EMBED_DIM,)  # one vector per volume, batch dim dropped
    assert torch.isfinite(embed).all()


def test_global_embed_handles_hf_decoded_nifti():
    # HF decodes niftis to a wrapper that breaks nibabel reorientation; a non-RAS volume must
    # still load. Same regression as the DLBS decode crash in test_unet.py.
    affine = np.diag([-1.0, -1.0, 1.0, 1.0])  # axcodes L,P,S -> forces reorientation
    dataset = Dataset.from_dict(
        {"image": [{"path": None, "bytes": make_image((28, 36, 30), affine).to_bytes()}]},
        features=Features({"image": Nifti()}),
    )
    wrapped = dataset[0]["image"]  # datasets Nifti1ImageWrapper, as the loader would yield
    assert make_model().global_embed(wrapped).shape == (EMBED_DIM,)


def test_transform_fits_grid_and_normalizes():
    sample = SmriMaeTransform(img_size=IMG_SIZE)(make_image((28, 36, 30)))
    image, mask = sample["image"], sample["mask"]

    assert image.shape == (1, *IMG_SIZE)  # padded up to the pretraining grid, channel first
    assert mask.shape == (1, *IMG_SIZE)
    assert image.dtype == torch.float32  # matches the weights outside the probe's autocast
    assert torch.equal(image == 0, ~mask)  # background zeroed, exactly where the mask is off
    brain = image[mask]
    assert brain.mean().abs() < 1e-4 and abs(brain.std().item() - 1.0) < 1e-3


def test_transform_resamples_to_1mm():
    # 2mm in-plane, 3mm through-plane: a 10x12x11-voxel box becomes 20x24x33 at 1mm.
    # Trilinear edges move the mean threshold by a voxel or so.
    sample = SmriMaeTransform(img_size=(64, 64, 64))(
        make_image((20, 24, 22), np.diag([2, 2, 3, 1]))
    )
    inside = sample["mask"][0].nonzero()
    extent = (inside.max(0).values - inside.min(0).values + 1).tolist()
    assert all(abs(size - target) <= 2 for size, target in zip(extent, (20, 24, 33)))


def test_transform_crops_volumes_larger_than_the_grid():
    sample = SmriMaeTransform(img_size=IMG_SIZE)(make_image((48, 60, 40)))
    assert sample["image"].shape == (1, *IMG_SIZE)


def test_resolve_ckpt_passes_through_local_paths(tmp_path):
    path = str(tmp_path / "checkpoint-last.pth")
    assert resolve_ckpt(path) == path
