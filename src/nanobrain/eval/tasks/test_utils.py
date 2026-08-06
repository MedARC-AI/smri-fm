import numpy as np
import pytest

from nanobrain.eval.tasks.utils import matched_indices


def cell_counts(cells: list, labels: list[int], keep: list[int]) -> dict:
    counts: dict = {}
    for index in keep:
        counts.setdefault(labels[index], {}).setdefault(cells[index], 0)
        counts[labels[index]][cells[index]] += 1
    return counts


def test_rejects_labels_outside_zero_one():
    with pytest.raises(AssertionError, match="0/1"):
        matched_indices(["a", "a", "a"], [0, 1, 2], seed=0)


def test_rejects_mismatched_cells_and_labels():
    with pytest.raises(ValueError):
        matched_indices(["a", "a"], [0, 1, 1], seed=0)


def test_classes_have_identical_cell_makeup():
    rng = np.random.default_rng(0)
    cells = rng.choice(["a", "b", "c"], 200).tolist()
    labels = rng.integers(0, 2, 200).tolist()
    keep = matched_indices(cells, labels, seed=0)
    counts = cell_counts(cells, labels, keep)
    assert counts[0] == counts[1]


def test_keeps_every_pairable_subject_when_uncapped():
    cells = ["a", "a", "a", "b", "b"]
    labels = [0, 0, 1, 0, 1]
    # cell 'a' pairs 1 of 2 negatives with its 1 positive, cell 'b' pairs its 1 and 1.
    assert len(matched_indices(cells, labels, seed=0)) == 4


def test_cap_drops_whole_pairs_and_stays_balanced():
    cells = ["a"] * 20
    labels = [0] * 10 + [1] * 10
    keep = matched_indices(cells, labels, seed=0, cap=6)
    assert len(keep) == 6
    assert sum(labels[i] for i in keep) == 3


def test_cell_with_one_class_is_dropped_entirely():
    cells = ["a", "a", "b", "b"]
    labels = [0, 1, 0, 0]
    keep = matched_indices(cells, labels, seed=0)
    assert [cells[i] for i in keep] == ["a", "a"]


def test_is_deterministic_across_calls_and_seed_dependent():
    rng = np.random.default_rng(1)
    cells = rng.choice(["a", "b"], 100).tolist()
    labels = rng.integers(0, 2, 100).tolist()
    assert matched_indices(cells, labels, seed=3) == matched_indices(cells, labels, seed=3)
    assert matched_indices(cells, labels, seed=3, cap=20) != matched_indices(
        cells, labels, seed=4, cap=20
    )


def test_tuple_cells_match_on_every_component():
    cells = [(s, m) for s in ("x", "y") for m in (0, 1) for _ in range(10)]
    labels = [i % 2 for i in range(len(cells))]
    keep = matched_indices(cells, labels, seed=0)
    counts = cell_counts(cells, labels, keep)
    assert counts[0] == counts[1]


@pytest.mark.parametrize("cap", [0, 2, 50])
def test_cap_never_exceeded(cap):
    cells = ["a"] * 40
    labels = [0] * 20 + [1] * 20
    assert len(matched_indices(cells, labels, seed=0, cap=cap)) <= max(cap, 0)
