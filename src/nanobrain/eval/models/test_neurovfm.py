"""NeuroVFM needs the optional `neurovfm` extra; these skip without it.

Only the preprocessing is checked -- against the upstream pipeline it mirrors, which is the
part we reimplemented and can get quietly wrong.
"""

import nibabel as nib
import numpy as np
import pytest
import torch

pytest.importorskip("neurovfm")

from nanobrain.eval.models.neurovfm import preprocess, to_sitk  # noqa: E402


def make_image(shape: tuple[int, int, int], affine: np.ndarray) -> nib.Nifti1Image:
    data = np.random.default_rng(0).random(shape, dtype=np.float32) * 1000
    return nib.Nifti1Image(data, affine)


@pytest.mark.parametrize(
    "shape, affine",
    [
        ((80, 96, 64), np.diag([1.0, 1.0, 3.0, 1.0])),  # anisotropic: resamples through-plane
        ((96, 112, 48), np.diag([-1.0, -1.0, 2.0, 1.0])),  # L,P,S -> forces reorientation
    ],
)
def test_preprocess_matches_upstream(tmp_path, shape, affine):
    from neurovfm.pipelines.preprocessor import StudyPreprocessor

    path = tmp_path / "image.nii.gz"
    nib.save(make_image(shape, affine), str(path))

    preproc = StudyPreprocessor()
    upstream = preproc.load_study(str(path), modality="mri")
    ours = preprocess(preproc, nib.load(str(path)))

    assert ours["img"].shape == upstream["img"].shape
    assert torch.equal(ours["coords"], upstream["coords"])
    assert torch.allclose(ours["img"], upstream["img"], atol=1e-4)


def test_to_sitk_roundtrips_geometry(tmp_path):
    import SimpleITK as sitk

    affine = np.array(
        [[0.0, -1.2, 0.0, 30.0], [0.9, 0.0, 0.0, -20.0], [0.0, 0.0, 3.0, 5.0], [0.0, 0.0, 0.0, 1.0]]
    )
    path = tmp_path / "image.nii.gz"
    nib.save(make_image((40, 48, 24), affine), str(path))

    ref = sitk.ReadImage(str(path))
    ours = to_sitk(nib.load(str(path)))

    assert np.allclose(ours.GetSpacing(), ref.GetSpacing())
    assert np.allclose(ours.GetDirection(), ref.GetDirection())
    assert np.allclose(ours.GetOrigin(), ref.GetOrigin())
    assert np.allclose(sitk.GetArrayFromImage(ours), sitk.GetArrayFromImage(ref))
