"""Neuro-JEPA needs the optional `neurojepa` extra; these skip without it.

The weights are gated, so only the preprocessing is checked here -- against the upstream
pipeline it mirrors, which is the part we reimplemented and can get quietly wrong.
"""

from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest
import torch

pytest.importorskip("monai")
pytest.importorskip("neurojepa")

from nanobrain.eval.models.neurojepa import (  # noqa: E402
    IMG_SIZE,
    _preprocess,
    _static_transform,
)


def _image(shape: tuple[int, int, int], affine: np.ndarray) -> nib.Nifti1Image:
    data = np.random.default_rng(0).random(shape, dtype=np.float32) * 1000
    return nib.Nifti1Image(data, affine)


def _upstream(path: str) -> torch.Tensor:
    """The fork's own static pipeline, reading from disk as its callers do."""
    from neurojepa.data.transforms import loading_transforms, vit3d_transforms

    sample = loading_transforms(roi=list(IMG_SIZE), spacing=(1.0, 1.0, 1.0), model_name="vit")(
        {"image": path}
    )
    cfg = SimpleNamespace(data=SimpleNamespace(img_size=list(IMG_SIZE)))
    return vit3d_transforms(cfg, mode="test")(sample)["image"]


@pytest.mark.parametrize(
    "shape, affine",
    [
        ((60, 70, 65), np.diag([1.0, 1.2, 1.0, 1.0])),  # off-1mm on one axis: resample, cheaply
        ((120, 140, 130), np.diag([-1.0, -1.0, 1.0, 1.0])),  # L,P,S -> forces reorientation
    ],
)
def test_preprocess_matches_upstream(tmp_path, shape, affine):
    path = str(tmp_path / "image.nii.gz")
    nib.save(_image(shape, affine), path)

    ours = _preprocess(_static_transform(), nib.load(path))
    assert ours.shape == (1, 1, *IMG_SIZE)
    assert torch.allclose(ours[0], _upstream(path).as_tensor(), atol=1e-4)
