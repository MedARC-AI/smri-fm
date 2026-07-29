"""SynthSeg needs the optional `synthseg` extra; these skip without it.

The weights are a separate download, so everything here exercises preprocessing alone.
"""

import nibabel as nib
import numpy as np
import pytest
import torch

pytest.importorskip("SynthSeg_pytorch")

from nanobrain.eval.models.synthseg import (  # noqa: E402
    MIN_SIZE,
    STRIDE,
    _bottleneck_box,
    _preprocess,
    _resample_torch,
)

CASES = [
    ((110, 120, 104), np.diag([1.0, 1.0, 1.0, 1.0])),  # already 1mm: no resample
    ((90, 100, 86), np.diag([1.33, 1.33, 1.33, 1.0])),  # coarser: upsample, no pre-blur
    ((140, 150, 60), np.diag([0.8, 0.8, 3.0, 1.0])),  # anisotropic: pre-blur on two axes
    ((100, 110, 96), np.diag([-1.2, -1.2, 1.2, 1.0])),  # L,P,S: reorientation plus resample
    ((96, 104, 90), np.array([[-1.0, 0, 0, 40], [0, 0, 1.0, -50], [0, -1.0, 0, 45], [0, 0, 0, 1]])),
]


def _image(shape, affine) -> nib.Nifti1Image:
    """A bright blob in the middle of an otherwise dark field."""
    rng = np.random.default_rng(0)
    data = rng.random(shape, dtype=np.float32) * 20
    box = tuple(slice(int(0.25 * s), int(0.75 * s)) for s in shape)
    data[box] += 400
    return nib.Nifti1Image(data, affine)


@pytest.mark.parametrize("shape, affine", CASES)
def test_verbatim_preprocess_matches_upstream(shape, affine):
    """Our chain must be SynthSeg's own preprocess, voxel for voxel."""
    from SynthSeg_pytorch.preprocessing import preprocess

    img = _image(shape, affine)
    ours, pad_idx = _preprocess(img, torch.device("cpu"), "verbatim")
    theirs, _, _, _, _, their_pad_idx, _ = preprocess(img, crop=None, min_pad=MIN_SIZE)

    np.testing.assert_array_equal(ours.numpy(), theirs[0, 0].astype(np.float32))
    np.testing.assert_array_equal(pad_idx, their_pad_idx)


@pytest.mark.parametrize("shape, affine", CASES)
def test_torch_resample_matches_the_reference_resample(shape, affine):
    """Same sample coordinates and same edge clamping, so only the arithmetic differs."""
    from SynthSeg_pytorch.preprocessing import resample_volume

    volume = np.asarray(_image(shape, affine).dataobj, dtype=np.float64)
    theirs, their_affine = resample_volume(volume, affine, [1.0, 1.0, 1.0])
    ours, our_affine = _resample_torch(volume, affine, torch.device("cpu"))

    np.testing.assert_allclose(our_affine, their_affine)
    assert ours.shape == theirs.shape
    # Their interpolation runs in float64, ours in float32, so scale the tolerance to the data.
    np.testing.assert_allclose(ours, theirs, atol=1e-5 * np.ptp(theirs), rtol=0)


@pytest.mark.parametrize("shape, affine", CASES)
def test_preprocess_output_is_a_valid_network_input(shape, affine):
    volume, pad_idx = _preprocess(_image(shape, affine), torch.device("cpu"), "torch")

    assert volume.ndim == 3
    for size in volume.shape:
        assert size % 32 == 0
        assert size >= MIN_SIZE
    assert 0.0 <= volume.min() <= volume.max() <= 1.0
    assert volume.max() > 0.9

    for axis, sl in enumerate(_bottleneck_box(pad_idx)):
        assert 0 <= sl.start < sl.stop <= volume.shape[axis] // STRIDE


def test_pooling_box_excludes_the_padding():
    """A scan padded up to the 128 floor must pool over the scan, not the padding around it."""
    volume, pad_idx = _preprocess(
        _image((70, 80, 90), np.diag([1.0, 1.0, 1.0, 1.0])), torch.device("cpu"), "torch"
    )

    assert tuple(volume.shape) == (MIN_SIZE, MIN_SIZE, MIN_SIZE)
    np.testing.assert_array_equal(pad_idx, [29, 24, 19, 99, 104, 109])
    # Cell 1 spans voxels [16, 32), which reaches into the padding; cell 2 is the first inside.
    assert _bottleneck_box(pad_idx) == (slice(2, 6), slice(2, 6), slice(2, 6))
    for axis, sl in enumerate(_bottleneck_box(pad_idx)):
        assert sl.start * STRIDE >= pad_idx[axis]
        assert sl.stop * STRIDE <= pad_idx[axis + 3]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("shape, affine", CASES)
def test_torch_resample_on_gpu_matches_cpu(shape, affine):
    img = _image(shape, affine)
    # cuDNN convolves in TF32 by default, which costs ~3 decimal digits in the blur; pin it off
    # so this measures the preprocessing rather than the backend's precision policy.
    with torch.backends.cudnn.flags(allow_tf32=False):
        on_cpu, _ = _preprocess(img, torch.device("cpu"), "torch")
        on_gpu, _ = _preprocess(img, torch.device("cuda"), "torch")

    assert on_cpu.shape == on_gpu.shape
    assert torch.allclose(on_cpu, on_gpu.cpu(), atol=1e-5)
