"""The wrapper around the vendored `smri_mae` package.

The pretrained checkpoints are too big to test against, so these cover the parts we wrote -- the
transform, the pooling contract, and checkpoint path resolution -- on a randomly-initialized
depth-0 encoder. Depth 0 keeps the tokenizing, masking and pooling while skipping the blocks:
their jagged SDPA asks the cuDNN backend whether it can run, which errors out on a machine with
a CUDA-built torch but no device.
"""

import itertools

import nibabel as nib
import numpy as np
import pytest
import torch
from torch import Tensor
from datasets import Dataset, Features, Nifti
from einops import rearrange
from nibabel.affines import apply_affine

import smri_mae.model_mae as models_mae
from nanobrain.eval.models.smri_mae import (
    SmriMae,
    SmriMaeTransform,
    fit_to_shape,
    rescale,
    resolve_ckpt,
)
from nanobrain.eval.nifti import canonical_img
from nanobrain.eval.probe_seg import seg_probe
from nanobrain.eval.tasks.base import SegmentationTask

IMG_SIZE = (32, 40, 32)
EMBED_DIM = 32
PATCH = 8
CPU = torch.device("cpu")


def make_model() -> SmriMae:
    mae = models_mae.MaskedAutoencoderViT(
        img_size=IMG_SIZE,
        patch_size=PATCH,
        depth=0,
        embed_dim=EMBED_DIM,
        num_heads=2,
        decoder_depth=0,
        decoder_embed_dim=EMBED_DIM,
        decoder_num_heads=2,
    )
    transform = SmriMaeTransform(img_size=IMG_SIZE)
    return SmriMae(mae.encoder, transform).eval()


def make_image(shape: tuple[int, int, int], affine: np.ndarray = np.eye(4)) -> nib.Nifti1Image:
    """A bright box in a noisy background, at raw MRI intensities the transform normalizes away.

    The box is half the volume along each axis, so the mean threshold recovers it exactly and
    its extent tracks the resampling.
    """
    data = np.random.default_rng(0).random(shape, dtype=np.float32) * 50
    box = tuple(slice(size // 4, size // 4 + size // 2) for size in shape)
    data[box] += 800
    return nib.Nifti1Image(data, affine)


def offset_affine(zooms: tuple[float, float, float]) -> np.ndarray:
    """Off the world origin, so an offset dropped anywhere in the chain cannot cancel out."""
    affine = np.diag([*zooms, 1.0])
    affine[:3, 3] = [-31.0, -44.0, -17.0]
    return affine


def test_global_embed_contract():
    embed = make_model().global_embed(make_image((28, 36, 30)))
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


def test_patch_embed_contract():
    model = make_model()
    img = make_image((28, 36, 30))
    sample = model.transform(img)
    features, coords = model.patch_embed(img)

    # one token per patch holding brain -- the ones `global_embed` pools, not the whole 4x5x4 grid
    blocks = rearrange(
        sample["mask"][0], "(x p) (y q) (z r) -> (x y z) (p q r)", p=PATCH, q=PATCH, r=PATCH
    )
    assert features.shape == (int(blocks.any(-1).sum()), EMBED_DIM)
    assert coords.shape == (len(features), 3)
    assert features.device.type == "cpu" and coords.device.type == "cpu"
    assert torch.isfinite(features).all() and features.std() > 0

    # the patches sit on the scan, not somewhere else in world space
    canon = canonical_img(img)
    corners = np.array(list(itertools.product(*[(0, size - 1) for size in canon.shape])))
    world = apply_affine(canon.affine, corners)
    assert (coords.numpy() >= world.min(axis=0) - PATCH).all()
    assert (coords.numpy() <= world.max(axis=0) + PATCH).all()


# Between them these cover both branches of `fit_to_shape`, with and without the 1mm rescale,
# including odd pads and a non-RAS scan that has to be reoriented first.
GEOMETRY_CASES = [
    ((27, 35, 31), offset_affine((1.0, 1.0, 1.0))),  # odd pads, no rescale
    ((48, 60, 40), offset_affine((1.0, 1.0, 1.0))),  # crop, no rescale
    ((14, 18, 20), offset_affine((2.0, 2.0, 1.5))),  # anisotropic rescale, then pad
    ((30, 24, 26), offset_affine((1.5, 2.5, 2.0))),  # anisotropic rescale, then odd crop
    ((28, 36, 30), np.diag([-1.0, -1.5, 1.0, 1.0])),  # axcodes L,P,S -> forces reorientation
]


def fit_geometry(transform: SmriMaeTransform, img: nib.Nifti1Image) -> tuple[Tensor, np.ndarray]:
    """`SmriMaeTransform.__call__`'s geometry alone; its z-score would corrupt a coordinate ramp.

    `test_fitted_affine_matches_the_pushed_coordinates` pins this against the affine the transform
    itself returns, so the two cannot drift apart.
    """
    canon = canonical_img(img)
    data = torch.from_numpy(np.ascontiguousarray(canon.get_fdata(dtype=np.float32)))
    affine = np.asarray(canon.affine, dtype=float)
    spacing = canon.header.get_zooms()
    if max(abs(s - s_) for s, s_ in zip(spacing, transform.spacing)) > 0.05:
        data, affine = rescale(data, affine, spacing, target_spacing=transform.spacing)
    return fit_to_shape(data, affine, transform.img_size)


def push_world_coords(transform: SmriMaeTransform, img: nib.Nifti1Image) -> np.ndarray:
    """Every fitted voxel's world coordinate, carried through as data rather than derived.

    Trilinear interpolation reproduces a linear ramp exactly, so pushing the three world-coordinate
    volumes through the resample and the shape fit reports where each output voxel came from --
    with no reasoning about `align_corners` conventions or which way the pad offsets point.
    """
    canon = canonical_img(img)
    voxels = rearrange(np.indices(canon.shape), "c x y z -> x y z c")
    world = apply_affine(canon.affine, voxels).astype(np.float32)
    ramps = [
        fit_geometry(transform, nib.Nifti1Image(world[..., ax], canon.affine))[0] for ax in range(3)
    ]
    return torch.stack(ramps, dim=-1).numpy()


@pytest.mark.parametrize("shape, affine", GEOMETRY_CASES)
def test_fitted_affine_matches_the_pushed_coordinates(shape, affine):
    transform = SmriMaeTransform(img_size=IMG_SIZE)
    img = make_image(shape, affine)
    volume, world_affine = fit_geometry(transform, img)
    assert volume.shape == IMG_SIZE
    # the helper duplicates the transform's composition, so pin it against the shipped one
    assert np.allclose(world_affine, transform(img)["affine"].numpy())

    voxels = rearrange(np.indices(volume.shape), "c x y z -> x y z c")
    predicted = apply_affine(world_affine, voxels)

    # padded voxels hold zero and the resample clamps at the scan border, so score the interior
    canon = canonical_img(img)
    source = apply_affine(np.linalg.inv(canon.affine), predicted)
    inside = ((source >= 1) & (source <= np.array(canon.shape) - 2)).all(axis=-1)
    assert inside.mean() > 0.1, "the fitted grid barely overlaps the scan"
    # a linear ramp survives trilinear interpolation exactly, so this is float32 rounding, not slack
    assert np.abs(push_world_coords(transform, img) - predicted)[inside].max() < 1e-3


def test_patch_centres_are_the_blocks_they_came_from():
    """Each returned centre must be the mean world position of the voxels `patchify3d` actually
    put in that token.

    Both sides are derived from the vendored patchifier and the brain mask rather than from a copy
    of `patch_embed`'s own arithmetic, so this pins the flattening order, the half-patch centre
    offset, and the `patch_ids` mapping back to the grid -- exactly to the millimetre, which a
    marker test can only bound to within a patch.
    """
    model = make_model()
    img = make_image((26, 22, 34), offset_affine((1.5, 2.0, 1.0)))
    _features, coords = model.patch_embed(img)

    sample = model.transform(img)
    world_affine = sample["affine"].numpy()
    voxels = rearrange(np.indices(IMG_SIZE), "c x y z -> x y z c")
    ramps = torch.from_numpy(apply_affine(world_affine, voxels)).float()  # (X, Y, Z, 3)
    blocks = torch.stack(
        [model.encoder.patchify(ramps[None, None, ..., ax]).mean(-1)[0] for ax in range(3)], dim=-1
    )  # (80, 3) in grid order, since a block's mean coordinate is its centre
    kept = rearrange(
        sample["mask"][0], "(x p) (y q) (z r) -> (x y z) (p q r)", p=PATCH, q=PATCH, r=PATCH
    ).any(-1)

    assert torch.allclose(coords, blocks[kept], atol=1e-2)


def test_patch_coords_land_on_the_marked_token():
    """A bright marker must change the token whose world centre covers it.

    Run through `patch_embed` itself, so the flattening order and the `patch_ids` mapping are under
    test alongside the affine. That only reads as a position test because this encoder is depth 0:
    self-attention is global, so on the real checkpoint the most-changed *output* token is
    unrelated to the marker and the test would pass while proving nothing.
    """
    model = make_model()
    affine = offset_affine((1.5, 2.0, 1.0))  # anisotropic, so the rescale is exercised
    base = np.random.default_rng(0).random((26, 22, 34), dtype=np.float32) * 50
    base[4:20, 6:14, 9:30] += 800  # asymmetric and off centre, so a flip or swap cannot pass

    reference, coords = model.patch_embed(nib.Nifti1Image(base, affine))
    assert len(coords) < np.prod(model.encoder.patchify.grid_size)  # empty patches really dropped
    truth_affine = canonical_img(nib.Nifti1Image(base, affine)).affine
    for voxel in [(8, 8, 12), (16, 11, 25)]:
        marked = base.copy()
        marked[tuple(slice(v - 1, v + 2) for v in voxel)] += 4000
        features, marked_coords = model.patch_embed(nib.Nifti1Image(marked, affine))
        assert torch.equal(marked_coords, coords)  # same patches kept, so the rows pair up
        hit = int((features - reference).norm(dim=1).argmax())
        truth = apply_affine(truth_affine, np.array(voxel))
        assert np.linalg.norm(coords[hit].numpy() - truth) < 1.5 * PATCH


def make_seg_subject(seed: int, with_lesion: bool) -> dict:
    """A noisy volume with an off-centre bright block, labelled in the seg. Anisotropic 1.2mm
    spacing so the probe only lines up if the rescale and shape fit are both in the affine."""
    shape, affine = (26, 30, 28), offset_affine((1.2, 1.2, 1.2))
    image = np.random.default_rng(seed).uniform(1.0, 2.0, shape).astype(np.float32)
    seg = np.zeros(shape, dtype=np.uint8)
    if with_lesion:
        box = (slice(3, 19), slice(5, 21), slice(8, 24))
        image[box], seg[box] = 2.5, 1
    return {
        col: {"path": None, "bytes": nib.Nifti1Image(arr, affine).to_bytes()}
        for col, arr in (("image", image), ("seg", seg))
    }


def test_seg_probe_runs_on_patch_embeddings():
    rows = [make_seg_subject(i, with_lesion=True) for i in range(4)]
    rows += [make_seg_subject(i + 100, with_lesion=False) for i in range(2)]
    dataset = Dataset.from_dict(
        {col: [row[col] for row in rows] for col in ("image", "seg")},
        features=Features({"image": Nifti(), "seg": Nifti()}),
    )
    task = SegmentationTask(
        name="fake", dataset_fn=lambda: dataset, seg_col="seg", class_names=("lesion",)
    )
    scores = seg_probe(
        make_model(), task,
        n_splits=3, n_repeats=1, seed=0, device=CPU, n_boot=100,
    )  # fmt: skip
    # 8mm patches over a 19mm block cap Dice at 0.86 even with oracle patch labels, and the
    # weights are random, so this is well short of perfect but far clear of chance
    assert scores["dice_lesion"] > 0.5
    assert scores["voxel_ap_lesion"] > 0.8


def test_resolve_ckpt_passes_through_local_paths(tmp_path):
    path = str(tmp_path / "checkpoint-last.pth")
    assert resolve_ckpt(path) == path
