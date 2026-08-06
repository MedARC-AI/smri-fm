"""NeuroVFM needs the optional `neurovfm` extra; these skip without it.

The preprocessing is checked against the upstream pipeline it mirrors, and the world coordinates
-- which upstream has no equivalent of -- against the content of the tokens they claim to place.
"""

import nibabel as nib
import numpy as np
import pytest
import torch
from datasets import Dataset, Features, Nifti
from nibabel.affines import apply_affine

pytest.importorskip("neurovfm")

from nanobrain.eval.models import create_model  # noqa: E402
from nanobrain.eval.models.neurovfm import from_sitk_affine, preprocess, to_sitk  # noqa: E402
from nanobrain.eval.nifti import canonical_img  # noqa: E402
from nanobrain.eval.probe_seg import seg_probe  # noqa: E402
from nanobrain.eval.tasks.base import SegmentationTask  # noqa: E402

CPU = torch.device("cpu")

# A token is 4 slices of 4mm by 16x16 in-plane millimetres, so a 16mm cube of the original scan.
TOKEN_MM = 16.0

BRIGHT = 1200.0

# Anisotropic and offset, with the 4mm axis on each of the three world axes in turn: that is what
# picks the branch in `transpose_to_dhw`, and each branch orders the array axes differently.
GEOMETRIES = [
    ((104, 124, 44), np.diag([1.1, 0.9, 3.0, 1.0]) + np.outer([-55, -70, -40, 0], [0, 0, 0, 1])),
    ((104, 40, 124), np.diag([1.1, 3.5, 0.9, 1.0]) + np.outer([-55, -70, -40, 0], [0, 0, 0, 1])),
    ((36, 124, 104), np.diag([4.0, 0.9, 1.1, 1.0]) + np.outer([-55, -70, -40, 0], [0, 0, 0, 1])),
]


@pytest.fixture(scope="module")
def model():
    return create_model("neurovfm")


def make_image(shape: tuple[int, int, int], affine: np.ndarray) -> nib.Nifti1Image:
    data = np.random.default_rng(0).random(shape, dtype=np.float32) * 1000
    return nib.Nifti1Image(data, affine)


def box_slices(
    shape: tuple[int, int, int], affine: np.ndarray, centre: tuple[float, float, float], mm: float
) -> tuple[slice, ...]:
    """The voxel box of half-width `mm` about `centre`, given as a fraction of the shape."""
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    half = np.maximum(1, np.round(mm / spacing)).astype(int)
    voxel = np.round(np.array(centre) * shape).astype(int)
    return tuple(slice(v - h, v + h + 1) for v, h in zip(voxel, half))


def make_phantom(
    shape: tuple[int, int, int],
    affine: np.ndarray,
    box: tuple[slice, ...] | None = None,
    seed: int = 0,
) -> nib.Nifti1Image:
    """Bright noise with an L-shaped dark corner, plus an optional box at the maximum intensity.

    NeuroVFM drops any token holding a voxel below the volume's 10th percentile, so that
    threshold has to land inside a contiguous dark region or every token is dropped. The corner
    is also what makes the phantom asymmetric: it is a different depth along each axis, so no
    axis swap or flip maps it onto itself.
    """
    data = np.random.default_rng(seed).uniform(800.0, BRIGHT, shape).astype(np.float32)
    data[: shape[0] // 7] = 0.0
    data[:, : shape[1] // 12] = 0.0
    if box is not None:
        data[box] = BRIGHT
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


def test_from_sitk_affine_matches_sitk():
    """The hand-built index -> world affine against SimpleITK's own index transform, on an
    oblique image so the direction cosines carry more than a permutation."""
    theta = 0.3
    affine = np.eye(4)
    affine[:2, :2] = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    affine[:3, :3] *= [1.1, 0.9, 3.0]
    affine[:3, 3] = [-55.0, -70.0, -40.0]

    img_sitk = to_sitk(make_phantom((104, 124, 44), affine))
    index = np.array([[0.0, 0.0, 0.0], [7.5, 3.0, 11.5], [39.0, 47.0, 23.0]])

    lps = np.array([img_sitk.TransformContinuousIndexToPhysicalPoint(i) for i in index.tolist()])
    assert np.allclose(apply_affine(from_sitk_affine(img_sitk), index), lps * [-1, -1, 1])


@pytest.mark.parametrize("shape, affine", GEOMETRIES)
@pytest.mark.parametrize("centre", [(0.35, 0.45, 0.25), (0.75, 0.8, 0.7), (0.45, 0.85, 0.15)])
def test_patch_coords_land_on_the_marked_token(model, shape, affine, centre):
    """A marker box must change the token whose world centre covers it.

    Checked against the raw tokenization rather than the backbone's output: self-attention is
    global, so the most-changed *output* token is unrelated to the marker's position and the test
    would pass while saying nothing. The marker sits at the volume's existing maximum so it leaves
    the min-max normalization alone, which would otherwise move every token at once.
    """
    box = box_slices(shape, affine, centre, mm=6.0)
    reference = preprocess(model.preproc, make_phantom(shape, affine))
    marked = preprocess(model.preproc, make_phantom(shape, affine, box))

    before = {tuple(c): t for c, t in zip(reference["coords"].tolist(), reference["img"])}
    after = zip(marked["coords"].tolist(), marked["img"])
    changed = np.array([float((t - before[tuple(c)]).norm()) for c, t in after])
    hit = int(changed.argmax())
    want = apply_affine(affine, [(s.start + s.stop - 1) / 2 for s in box])

    assert np.linalg.norm(marked["world"][hit].numpy() - want) < TOKEN_MM
    assert changed.max() > 10 * np.median(changed)  # the marker, not a global rescale


@pytest.mark.parametrize("shape, affine", GEOMETRIES)
def test_patch_coords_round_trip_onto_the_brain(model, shape, affine):
    """Every patch centre must land inside the input volume, on its bright region."""
    img = make_phantom(shape, affine)
    _features, coords = model.patch_embed(img)

    canonical = canonical_img(img)
    voxels = apply_affine(np.linalg.inv(canonical.affine), coords.numpy())
    assert (voxels > -0.5).all() and (voxels < np.array(canonical.shape) - 0.5).all()

    data = canonical.get_fdata()
    nearest = tuple(np.round(voxels).astype(int).T)
    assert (data[nearest] > BRIGHT / 2).all()  # the dark corner is dropped as background


def test_seg_probe_runs_end_to_end(model):
    """The whole probe over synthetic subjects: patch coverage, then a resolvable structure."""
    shape, affine = GEOMETRIES[0]
    box = box_slices(shape, affine, (0.6, 0.6, 0.5), mm=16.0)  # a lesion a couple of tokens wide
    rows = []
    for seed in range(6):
        seg = np.zeros(shape, dtype=np.uint8)
        seg[box] = 1
        rows.append((make_phantom(shape, affine, box, seed=seed), nib.Nifti1Image(seg, affine)))

    dataset = Dataset.from_dict(
        {
            "image": [{"path": None, "bytes": image.to_bytes()} for image, _ in rows],
            "seg": [{"path": None, "bytes": seg.to_bytes()} for _, seg in rows],
        },
        features=Features({"image": Nifti(), "seg": Nifti()}),
    )
    task = SegmentationTask(
        name="fake", dataset_fn=lambda: dataset, seg_col="seg", class_names=("lesion",)
    )
    scores = seg_probe(
        model, task,
        n_splits=3, n_repeats=1, seed=0, device=CPU, n_boot=100,
    )  # fmt: skip

    assert scores["dice_lesion"] > 0.5  # blunted by 16mm tokens against a 32mm structure
    assert scores["voxel_ap_lesion"] > 0.5
