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
    preprocess,
    static_transform,
)


def make_image(shape: tuple[int, int, int], affine: np.ndarray) -> nib.Nifti1Image:
    data = np.random.default_rng(0).random(shape, dtype=np.float32) * 1000
    return nib.Nifti1Image(data, affine)


def upstream_volume(path: str) -> torch.Tensor:
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
    nib.save(make_image(shape, affine), path)

    ours = preprocess(static_transform(), nib.load(path), torch.device("cpu"))
    assert ours.shape == (1, 1, *IMG_SIZE)
    assert torch.allclose(ours[0], upstream_volume(path).as_tensor(), atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_preprocess_on_gpu_matches_cpu(tmp_path):
    # The resample runs on GPU via cupy rather than scipy, so pin that it agrees with the CPU
    # chain the equivalence test above anchors.
    path = str(tmp_path / "image.nii.gz")
    nib.save(make_image((150, 180, 150), np.diag([1.33, 1.0, 1.0, 1.0])), path)
    img = nib.load(path)

    transform = static_transform()
    on_cpu = preprocess(transform, img, torch.device("cpu"))
    on_gpu = preprocess(transform, img, torch.device("cuda")).cpu()

    assert torch.allclose(on_cpu, on_gpu, atol=1e-3)
