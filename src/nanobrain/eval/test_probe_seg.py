import nibabel as nib
import numpy as np
import pytest
import torch
from datasets import Dataset, Features, Nifti
from einops import rearrange
from nibabel.affines import apply_affine

from nanobrain.eval import probe_seg
from nanobrain.eval.models.base import PatchFeatures
from nanobrain.eval.nifti import canonical, canonical_img
from nanobrain.eval.probe_seg import dice_score, score_prediction, seg_probe, select_voxels
from nanobrain.eval.tasks.base import SegmentationTask

CPU = torch.device("cpu")

# Anisotropic and offset, so a probe that confused voxel indices for world mm would not match up
AFFINE = np.array([[2.0, 0, 0, -30.0], [0, 3.0, 0, -40.0], [0, 0, 1.5, -20.0], [0, 0, 0, 1.0]])


class FakeSegModel:
    """One patch per voxel, placed at that voxel's world position; feature = [intensity, 0].
    Intensity encodes the class, so a linear head separates the classes deterministically."""

    def patch_embed(self, img: nib.Nifti1Image) -> PatchFeatures:
        image = canonical(img)
        voxels = np.indices(image.shape).reshape(3, -1).T
        coords = apply_affine(canonical_img(img).affine, voxels)
        features = torch.stack([image.reshape(-1), torch.zeros(image.numel())], dim=-1)
        return PatchFeatures(features, torch.from_numpy(coords).float())


class VoxelCoordModel(FakeSegModel):
    """Reports patch centres as voxel indices instead of world mm -- the affine bug to catch."""

    def patch_embed(self, img: nib.Nifti1Image) -> PatchFeatures:
        features, _ = super().patch_embed(img)
        voxels = np.indices(canonical(img).shape).reshape(3, -1).T
        return PatchFeatures(features, torch.from_numpy(voxels).float())


class TransposedCoordModel(FakeSegModel):
    """Swaps two coordinate axes. On a cubic grid the point cloud is unchanged, so no geometric
    check can see it -- only the labels landing on the wrong features."""

    def patch_embed(self, img: nib.Nifti1Image) -> PatchFeatures:
        features, coords = super().patch_embed(img)
        return PatchFeatures(features, coords[:, [1, 0, 2]])


class ShortCoordModel(FakeSegModel):
    """Returns fewer coords than features, as a token-dropping backbone easily could."""

    def patch_embed(self, img: nib.Nifti1Image) -> PatchFeatures:
        features, coords = super().patch_embed(img)
        return PatchFeatures(features, coords[:-5])


class CoarseSegModel:
    """One patch per 4^3 block, at the block's world centre; feature = [mean intensity, 0].
    Many voxels per patch, which is the regime the per-voxel fakes never reach."""

    def patch_embed(self, img: nib.Nifti1Image) -> PatchFeatures:
        image = canonical(img)
        blocks = rearrange(image, "(x p) (y q) (z r) -> x y z (p q r)", p=4, q=4, r=4)
        means = blocks.mean(-1).reshape(-1)
        centres = np.indices(blocks.shape[:3]).reshape(3, -1).T * 4 + 1.5
        coords = apply_affine(canonical_img(img).affine, centres)
        features = torch.stack([means, torch.zeros_like(means)], dim=-1)
        return PatchFeatures(features, torch.from_numpy(coords).float())


def encode_nifti(arr: np.ndarray, affine: np.ndarray = AFFINE) -> dict:
    return {"path": None, "bytes": nib.Nifti1Image(arr, affine).to_bytes()}


# Placed so that swapping any two axes lands each box entirely on background, which is what makes
# a transposed coordinate mapping visible. Aligned to 4^3 blocks so a coarse model can resolve them.
BOXES = [
    (slice(4, 8), slice(8, 12), slice(8, 12)),
    (slice(8, 12), slice(0, 4), slice(0, 4)),
]


def make_subject(
    seed: int, blocks: dict[int, float], shape=(16, 16, 16), affine: np.ndarray = AFFINE
) -> dict:
    """A subject whose image has one high-intensity box per label in `blocks`, seg labelled."""
    rng = np.random.default_rng(seed)
    image = rng.uniform(1.0, 2.0, shape).astype(np.float32)  # in-brain background
    seg = np.zeros(shape, dtype=np.uint8)
    assert len(blocks) <= len(BOXES), "no box laid out for that many classes"
    for box, (label, intensity) in zip(BOXES, blocks.items()):
        image[box] = intensity
        seg[box] = label
    return {"image": encode_nifti(image, affine), "seg": encode_nifti(seg, affine)}


def make_dataset(rows: list[dict]) -> Dataset:
    return Dataset.from_dict(
        {"image": [r["image"] for r in rows], "seg": [r["seg"] for r in rows]},
        features=Features({"image": Nifti(), "seg": Nifti()}),
    )


def make_task(dataset: Dataset, class_names: tuple[str, ...]) -> SegmentationTask:
    return SegmentationTask(
        name="fake", dataset_fn=lambda: dataset, seg_col="seg", class_names=class_names
    )


def test_select_voxels_keeps_all_foreground_and_caps_background():
    labels = np.zeros(100, dtype=int)
    labels[:6] = 1  # 6 foreground voxels
    y = labels[select_voxels(labels, np.random.default_rng(0))]
    assert (y == 1).sum() == 6  # every foreground voxel kept
    assert (y == 0).sum() == min(94, probe_seg.NEG_PER_SUBJECT)


def test_dice_perfect_and_disjoint():
    a = np.array([True, True, False, False])
    assert dice_score(a, a) == 1.0
    assert dice_score(a, ~a) == 0.0


def test_score_present_and_empty_classes():
    # 3 voxels: labels [0, 1, 2]; probs put argmax on the right class for voxels 1 and 2.
    probs = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    dice, ap = score_prediction(np.array([0, 1, 2]), probs)
    assert dice.tolist() == [1.0, 1.0] and ap.tolist() == [1.0, 1.0]

    # class 2 absent from ground truth, and nothing predicted as 2 -> specificity 1, AP undefined.
    dice, ap = score_prediction(np.array([0, 1, 1]), probs[[0, 1, 1]])
    assert dice[1] == 1.0 and np.isnan(ap[1])
    # a false positive for the absent class drops its specificity to 0.
    dice, _ = score_prediction(np.array([0, 0, 0]), probs)
    assert dice[1] == 0.0


def test_seg_probe_detects_foreground():
    rows = [make_subject(i, {1: 8.0}) for i in range(4)]  # all carry a lesion
    rows += [make_subject(i + 100, {}) for i in range(2)]  # label-negative subjects
    dataset = make_dataset(rows)
    scores = seg_probe(
        FakeSegModel(), make_task(dataset, ("lesion",)),
        n_splits=3, n_repeats=1, seed=0, device=CPU, n_boot=200,
    )  # fmt: skip
    assert {"dice_lesion", "voxel_ap_lesion", "dice", "voxel_ap"} <= set(scores)
    for key in ("dice_lesion", "voxel_ap_lesion"):
        assert scores[f"{key}_ci_low"] <= scores[key] <= scores[f"{key}_ci_high"]
    assert scores["dice_lesion"] > 0.8
    assert scores["voxel_ap_lesion"] > 0.8


def test_seg_probe_multiclass():
    rows = [make_subject(i, {1: 6.0, 2: 12.0}) for i in range(6)]
    dataset = make_dataset(rows)
    scores = seg_probe(
        FakeSegModel(), make_task(dataset, ("nerve", "vessel")),
        n_splits=3, n_repeats=1, seed=0, device=CPU, n_boot=200,
    )  # fmt: skip
    assert {"dice_nerve", "dice_vessel", "voxel_ap_nerve", "voxel_ap_vessel"} <= set(scores)
    assert scores["dice"] > 0.7  # macro over both structures
    assert scores["dice_nerve"] > 0.7 and scores["dice_vessel"] > 0.7


def test_seg_probe_requires_present_class():
    dataset = make_dataset([make_subject(i, {}) for i in range(4)])  # no foreground anywhere
    with pytest.raises(AssertionError, match="absent"):
        seg_probe(
            FakeSegModel(), make_task(dataset, ("lesion",)),
            n_splits=2, n_repeats=1, seed=0, device=CPU, n_boot=50,
        )  # fmt: skip


def test_seg_probe_rejects_coords_off_the_brain():
    dataset = make_dataset([make_subject(i, {1: 8.0}) for i in range(4)])
    with pytest.raises(AssertionError, match="do not cover the brain"):
        seg_probe(
            VoxelCoordModel(), make_task(dataset, ("lesion",)),
            n_splits=2, n_repeats=1, seed=0, device=CPU, n_boot=50,
        )  # fmt: skip


def test_seg_probe_rejects_coords_not_matching_features():
    dataset = make_dataset([make_subject(i, {1: 8.0}) for i in range(4)])
    with pytest.raises(AssertionError, match="patch features against coords"):
        seg_probe(
            ShortCoordModel(), make_task(dataset, ("lesion",)),
            n_splits=2, n_repeats=1, seed=0, device=CPU, n_boot=50,
        )  # fmt: skip


def test_seg_probe_scores_badly_on_transposed_coords():
    # A cubic grid hides an axis swap from every geometric check, so the asymmetric fixture is
    # what catches it: the same features land on the wrong voxels and Dice collapses.
    rows = [make_subject(i, {1: 8.0}, affine=np.eye(4)) for i in range(4)]
    dataset = make_dataset(rows)
    task = make_task(dataset, ("lesion",))
    good = seg_probe(
        FakeSegModel(), task,
        n_splits=2, n_repeats=1, seed=0, device=CPU, n_boot=50,
    )  # fmt: skip
    bad = seg_probe(
        TransposedCoordModel(), task,
        n_splits=2, n_repeats=1, seed=0, device=CPU, n_boot=50,
    )  # fmt: skip
    assert good["dice_lesion"] > 0.8
    assert bad["dice_lesion"] < 0.3


def test_seg_probe_gathers_coarse_patches():
    # Each patch covers 64 voxels, so predictions are piecewise constant and must be gathered out.
    rows = [make_subject(i, {1: 8.0}) for i in range(4)]
    rows += [make_subject(i + 100, {}) for i in range(2)]
    dataset = make_dataset(rows)
    scores = seg_probe(
        CoarseSegModel(), make_task(dataset, ("lesion",)),
        n_splits=3, n_repeats=1, seed=0, device=CPU, n_boot=200,
    )  # fmt: skip
    assert scores["dice_lesion"] > 0.8  # the boxes are block-aligned, so this can be resolved
    assert scores["voxel_ap_lesion"] > 0.8
