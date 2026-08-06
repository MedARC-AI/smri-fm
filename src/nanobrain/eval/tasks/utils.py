"""Shared helpers for building task datasets."""

from collections import defaultdict

import numpy as np


def matched_indices(cells: list, labels: list[int], seed: int, cap: int | None = None) -> list[int]:
    """Indices of a subcohort in which both classes have an identical `cells` makeup.

    Each positive is paired with a negative drawn from the same cell, so anything constant
    within a cell -- site, scanner, age band, sex -- carries no information about the label.
    `cap` keeps at most that many subjects by dropping whole pairs, preserving the balance.
    """
    assert set(labels) <= {0, 1}, f"labels must be 0/1, got {sorted(set(labels))}"

    negatives: dict[object, list[int]] = defaultdict(list)
    positives: dict[object, list[int]] = defaultdict(list)
    for index, (cell, label) in enumerate(zip(cells, labels, strict=True)):
        bucket = positives if label == 1 else negatives
        bucket[cell].append(index)

    rng = np.random.default_rng(seed)
    pairs = []
    for cell in sorted(negatives.keys() | positives.keys()):
        negative, positive = negatives[cell], positives[cell]
        size = min(len(negative), len(positive))
        pairs += zip(
            rng.choice(negative, size, replace=False).tolist(),
            rng.choice(positive, size, replace=False).tolist(),
        )

    pairs = [pairs[i] for i in rng.permutation(len(pairs))]
    if cap is not None:
        pairs = pairs[: cap // 2]
    return sorted(index for pair in pairs for index in pair)
