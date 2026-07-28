"""Shared helpers for building task datasets."""

import numpy as np


def matched_indices(cells: list, labels: list[int], seed: int, cap: int | None = None) -> list[int]:
    """Indices of a subcohort in which both classes have an identical `cells` makeup.

    Each positive is paired with a negative drawn from the same cell, so anything constant
    within a cell -- site, scanner, age band, sex -- carries no information about the label.
    `cap` keeps at most that many subjects by dropping whole pairs, preserving the balance.
    """
    by_cell: dict = {}
    for index, (cell, label) in enumerate(zip(cells, labels)):
        by_cell.setdefault(cell, ([], []))[label].append(index)

    rng = np.random.default_rng(seed)
    pairs = []
    for cell in sorted(by_cell):
        negative, positive = by_cell[cell]
        size = min(len(negative), len(positive))
        pairs += zip(
            rng.choice(negative, size, replace=False).tolist(),
            rng.choice(positive, size, replace=False).tolist(),
        )

    pairs = [pairs[i] for i in rng.permutation(len(pairs))]
    if cap is not None:
        pairs = pairs[: cap // 2]
    return sorted(index for pair in pairs for index in pair)
