"""SynthSeg needs the optional `synthseg` extra; these skip without it.

Most of it exercises preprocessing alone. The two tests that need the network skip when the
weights, which are a separate download, are not cached.
"""

import os

import nibabel as nib
import numpy as np
import pytest
import torch
from nibabel.affines import apply_affine

pytest.importorskip("SynthSeg_pytorch")

from nanobrain.eval.models.synthseg import (  # noqa: E402
    MIN_SIZE,
    STRIDE,
    bottleneck_box,
    preprocess,
    resample_torch,
)
from nanobrain.eval.nifti import canonical_img  # noqa: E402
from SynthSeg_pytorch.predict import get_model_dir  # noqa: E402

needs_weights = pytest.mark.skipif(
    not os.path.exists(os.path.join(get_model_dir(), "synthseg_2.0.h5")),
    reason="needs the downloaded SynthSeg weights",
)

CASES = [
    ((110, 120, 104), np.diag([1.0, 1.0, 1.0, 1.0])),  # already 1mm: no resample
    ((90, 100, 86), np.diag([1.33, 1.33, 1.33, 1.0])),  # coarser: upsample, no pre-blur
    ((140, 150, 60), np.diag([0.8, 0.8, 3.0, 1.0])),  # anisotropic: pre-blur on two axes
    ((100, 110, 96), np.diag([-1.2, -1.2, 1.2, 1.0])),  # L,P,S: reorientation plus resample
    ((96, 104, 90), np.array([[-1.0, 0, 0, 40], [0, 0, 1.0, -50], [0, -1.0, 0, 45], [0, 0, 0, 1]])),
]


def make_image(shape, affine) -> nib.Nifti1Image:
    """A bright blob in the middle of an otherwise dark field."""
    rng = np.random.default_rng(0)
    data = rng.random(shape, dtype=np.float32) * 20
    box = tuple(slice(int(0.25 * s), int(0.75 * s)) for s in shape)
    data[box] += 400
    return nib.Nifti1Image(data, affine)


@pytest.mark.parametrize("shape, affine", CASES)
def test_verbatim_preprocess_matches_upstream(shape, affine):
    """Our chain must be SynthSeg's own preprocess, voxel for voxel."""
    from SynthSeg_pytorch.preprocessing import preprocess as upstream_preprocess

    img = make_image(shape, affine)
    ours, pad_idx, _world = preprocess(img, torch.device("cpu"), "verbatim")
    theirs, _, _, _, _, their_pad_idx, _ = upstream_preprocess(img, crop=None, min_pad=MIN_SIZE)

    np.testing.assert_array_equal(ours.numpy(), theirs[0, 0].astype(np.float32))
    np.testing.assert_array_equal(pad_idx, their_pad_idx)


@pytest.mark.parametrize("shape, affine", CASES)
def test_torch_resample_matches_the_reference_resample(shape, affine):
    """Same sample coordinates and same edge clamping, so only the arithmetic differs."""
    from SynthSeg_pytorch.preprocessing import resample_volume

    volume = np.asarray(make_image(shape, affine).dataobj, dtype=np.float64)
    theirs, their_affine = resample_volume(volume, affine, [1.0, 1.0, 1.0])
    ours, our_affine = resample_torch(volume, affine, torch.device("cpu"))

    np.testing.assert_allclose(our_affine, their_affine)
    assert ours.shape == theirs.shape
    # Their interpolation runs in float64, ours in float32, so scale the tolerance to the data.
    np.testing.assert_allclose(ours, theirs, atol=1e-5 * np.ptp(theirs), rtol=0)


@pytest.mark.parametrize("shape, affine", CASES)
def test_preprocess_output_is_a_valid_network_input(shape, affine):
    volume, pad_idx, _world = preprocess(make_image(shape, affine), torch.device("cpu"), "torch")

    assert volume.ndim == 3
    for size in volume.shape:
        assert size % 32 == 0
        assert size >= MIN_SIZE
    assert 0.0 <= volume.min() <= volume.max() <= 1.0
    assert volume.max() > 0.9

    for axis, sl in enumerate(bottleneck_box(pad_idx)):
        assert 0 <= sl.start < sl.stop <= volume.shape[axis] // STRIDE


def test_pooling_box_excludes_the_padding():
    """A scan padded up to the 128 floor must pool over the scan, not the padding around it."""
    volume, pad_idx, _world = preprocess(
        make_image((70, 80, 90), np.diag([1.0, 1.0, 1.0, 1.0])), torch.device("cpu"), "torch"
    )

    assert tuple(volume.shape) == (MIN_SIZE, MIN_SIZE, MIN_SIZE)
    np.testing.assert_array_equal(pad_idx, [29, 24, 19, 99, 104, 109])
    # Cell 1 spans voxels [16, 32), which reaches into the padding; cell 2 is the first inside.
    assert bottleneck_box(pad_idx) == (slice(2, 6), slice(2, 6), slice(2, 6))
    for axis, sl in enumerate(bottleneck_box(pad_idx)):
        assert sl.start * STRIDE >= pad_idx[axis]
        assert sl.stop * STRIDE <= pad_idx[axis + 3]


@pytest.mark.parametrize("resample", ["verbatim", "torch"])
@pytest.mark.parametrize("shape, affine", CASES)
def test_world_affine_tracks_the_whole_geometric_chain(monkeypatch, shape, affine, resample):
    """Feed the pipeline the world coordinates themselves and read them back out.

    Resampling, reorientation and padding are all intensity-independent, so an input holding one
    world coordinate per voxel comes out holding that same coordinate at wherever the voxel moved
    to -- which is exactly what the returned affine has to predict. Blur and edge clamping bend the
    ramp at the scan boundary, so the comparison skips a margin there.

    The percentile rescale would clip the coordinates it is handed, and moves no geometry.
    """
    import SynthSeg_pytorch.preprocessing as upstream

    monkeypatch.setattr(upstream, "rescale_volume", lambda volume, **kwargs: volume)
    margin = 4
    world = apply_affine(affine, np.indices(shape).reshape(3, -1).T)

    for axis in range(3):
        coordinate = world[:, axis].reshape(shape)
        img = nib.Nifti1Image(coordinate, affine)
        volume, pad_idx, world_affine = preprocess(img, torch.device("cpu"), resample)

        interior = np.indices(tuple(pad_idx[3:] - pad_idx[:3] - 2 * margin)).reshape(3, -1).T
        interior += pad_idx[:3] + margin
        stored = volume.numpy()[tuple(interior.T)]
        np.testing.assert_allclose(stored, apply_affine(world_affine, interior)[:, axis], atol=1e-3)


@needs_weights
def test_patch_coords_cover_the_canonical_grid():
    """Every cell the scan occupies must map back inside the grid the seg probe labels on."""
    from nanobrain.eval.models import create_model

    shape, affine = CASES[-1]
    img = make_image(shape, affine)
    features, coords = create_model("synthseg").patch_embed(img)

    canonical = canonical_img(img)
    assert features.shape == (len(coords), 384)
    voxels = apply_affine(np.linalg.inv(canonical.affine), coords.numpy())
    inside = np.all((voxels > -STRIDE) & (voxels < np.array(canonical.shape) + STRIDE), axis=1)
    # the padding pushes cells off the scan; what matters is that the scan itself is covered
    assert inside.sum() > 0.4 * len(coords)
    np.testing.assert_allclose(voxels[inside].min(0), 0, atol=STRIDE)
    np.testing.assert_allclose(voxels[inside].max(0), np.array(canonical.shape) - 1, atol=STRIDE)


@needs_weights
def test_patch_coords_locate_a_marker():
    """A dark blob must move the bottleneck features sitting over its own world position.

    Scored on the change-weighted centroid of the response rather than the single most-changed
    cell: five levels of 3x3x3 convolutions give the bottleneck a ~125-voxel receptive field, so
    the argmax cell wanders by tens of mm while the response as a whole stays on the marker. The
    affine here permutes and flips two axes, which is where a silent ordering bug would show.
    """
    from nanobrain.eval.models import create_model

    model = create_model("synthseg")
    shape = (100, 110, 96)
    affine = np.array([[-1.0, 0, 0, 60], [0, 0, 1.0, -50], [0, -1.0, 0, 45], [0, 0, 0, 1]])
    base = np.random.default_rng(0).uniform(10, 30, shape).astype(np.float32)
    base[tuple(slice(int(0.2 * s), int(0.8 * s)) for s in shape)] += 200  # a head-sized block

    reference, coords = model.patch_embed(nib.Nifti1Image(base, affine))
    markers = [(30, 35, 30), (70, 80, 65), (50, 55, 48)]
    centroids = []
    for voxel in markers:
        marked = base.copy()
        marked[tuple(slice(v - 6, v + 6) for v in voxel)] = 0.0
        features, _ = model.patch_embed(nib.Nifti1Image(marked, affine))
        change = (features - reference).norm(dim=1).numpy()
        centroids.append((coords.numpy() * change[:, None]).sum(0) / change.sum())

    truth = apply_affine(affine, np.array(markers, dtype=float))
    offsets = np.linalg.norm(np.array(centroids)[:, None] - truth[None], axis=-1)
    assert np.all(np.diag(offsets) < STRIDE), f"response centroids {offsets.diagonal().round(1)}mm"
    # and each response is nearer its own marker than any other, which no shrinkage can fake
    np.testing.assert_array_equal(offsets.argmin(axis=1), np.arange(len(markers)))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("shape, affine", CASES)
def test_torch_resample_on_gpu_matches_cpu(shape, affine):
    img = make_image(shape, affine)
    # cuDNN convolves in TF32 by default, which costs ~3 decimal digits in the blur; pin it off
    # so this measures the preprocessing rather than the backend's precision policy.
    with torch.backends.cudnn.flags(allow_tf32=False):
        on_cpu, _, world_cpu = preprocess(img, torch.device("cpu"), "torch")
        on_gpu, _, world_gpu = preprocess(img, torch.device("cuda"), "torch")

    assert on_cpu.shape == on_gpu.shape
    assert torch.allclose(on_cpu, on_gpu.cpu(), atol=1e-5)
    np.testing.assert_allclose(world_cpu, world_gpu)
