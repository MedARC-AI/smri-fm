import nibabel as nib
import numpy as np
import pytest
import torch
from datasets import Dataset, Features, Nifti

from nanobrain.eval import probe_seg
from nanobrain.eval.nifti import canonical
from nanobrain.eval.probe_seg import dice_score, score_prediction, seg_probe, subsample
from nanobrain.eval.tasks.base import SegmentationTask

CPU = torch.device("cpu")


class FakeSegModel:
    """dense_embed feature = [intensity, 0]; intensity encodes the class, so a linear head
    separates foreground from background (and classes from each other) deterministically."""

    def dense_embed(self, img: nib.Nifti1Image) -> torch.Tensor:
        image = canonical(img)  # (X, Y, Z)
        return torch.stack([image, torch.zeros_like(image)], dim=-1)  # (X, Y, Z, 2)


def encode_nifti(arr: np.ndarray) -> dict:
    return {"path": None, "bytes": nib.Nifti1Image(arr, np.eye(4)).to_bytes()}


def make_subject(seed: int, blocks: dict[int, float], shape=(16, 16, 16)) -> dict:
    """A subject whose image has a 4^3 high-intensity cube per label in `blocks`, seg labelled."""
    rng = np.random.default_rng(seed)
    image = rng.uniform(1.0, 2.0, shape).astype(np.float32)  # in-brain background
    seg = np.zeros(shape, dtype=np.uint8)
    for offset, (label, intensity) in enumerate(blocks.items()):
        lo = 2 + 5 * offset
        cube = (slice(lo, lo + 4),) * 3
        image[cube] = intensity
        seg[cube] = label
    return {"image": encode_nifti(image), "seg": encode_nifti(seg)}


def make_dataset(rows: list[dict]) -> Dataset:
    return Dataset.from_dict(
        {"image": [r["image"] for r in rows], "seg": [r["seg"] for r in rows]},
        features=Features({"image": Nifti(), "seg": Nifti()}),
    )


def make_task(dataset: Dataset, class_names: tuple[str, ...]) -> SegmentationTask:
    return SegmentationTask(
        name="fake", dataset_fn=lambda: dataset, seg_col="seg", class_names=class_names
    )


def test_subsample_keeps_all_foreground_and_caps_background():
    feats = np.arange(100).reshape(100, 1).astype(float)
    labels = np.zeros(100, dtype=int)
    labels[:6] = 1  # 6 foreground voxels
    mask = np.ones(100, dtype=bool)
    rng = np.random.default_rng(0)
    _, y = subsample(feats, labels, mask, rng)
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
        FakeSegModel(), make_task(dataset, ("lesion",)), dataset,
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
        FakeSegModel(), make_task(dataset, ("nerve", "vessel")), dataset,
        n_splits=3, n_repeats=1, seed=0, device=CPU, n_boot=200,
    )  # fmt: skip
    assert {"dice_nerve", "dice_vessel", "voxel_ap_nerve", "voxel_ap_vessel"} <= set(scores)
    assert scores["dice"] > 0.7  # macro over both structures
    assert scores["dice_nerve"] > 0.7 and scores["dice_vessel"] > 0.7


def test_seg_probe_requires_present_class():
    dataset = make_dataset([make_subject(i, {}) for i in range(4)])  # no foreground anywhere
    with pytest.raises(AssertionError, match="absent"):
        seg_probe(
            FakeSegModel(), make_task(dataset, ("lesion",)), dataset,
            n_splits=2, n_repeats=1, seed=0, device=CPU, n_boot=50,
        )  # fmt: skip
