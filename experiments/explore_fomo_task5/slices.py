"""One axial slice per subject, drawn at the full field of view, to show the AP clipping.

Axial puts the anterior-posterior axis in the plane, so a scan cut through the cerebrum ends in a
flat edge against the top or bottom of its own panel. Each panel is the whole slice with no
cropping, the brain outline in cyan, and the parts of it lying against the edge of the field of
view in red. Needs `segment.py` to have run.

    uv run python experiments/explore_fomo_task5/slices.py    # -> figures/clipping.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from segment import scanner, subjects

SCANNER_COLOR = {"Skyra": "tab:blue", "Cigna": "tab:orange"}


def panel(ax, t1: np.ndarray, brain: np.ndarray, title: str, color: str) -> None:
    """Axial slice at the widest part of the brain, anterior up, subject left on the left."""
    index = int(brain.sum(axis=(0, 1)).argmax())
    image, mask = t1[:, :, index].T, brain[:, :, index].T

    low, high = np.percentile(image, (1, 99.5))
    ax.imshow(image, cmap="gray", vmin=low, vmax=high, origin="lower", aspect="equal")
    ax.contour(mask, levels=[0.5], colors="cyan", linewidths=0.5)
    for row in (0, mask.shape[0] - 1):
        columns = np.flatnonzero(mask[row])
        if columns.size:
            ax.plot(columns, np.full(columns.size, row), color="red", linewidth=2.5)
    ax.set_title(title, fontsize=8, color=color)
    ax.set_xticks([])
    ax.set_yticks([])


def main() -> None:
    parser = argparse.ArgumentParser()
    here = Path(__file__).parent
    parser.add_argument("--out", type=Path, default=here / "output")
    parser.add_argument("--figures", type=Path, default=here / "figures")
    args = parser.parse_args()

    rows = [(s, y) for s, y in subjects() if (args.out / s / "seg.nii.gz").exists()]
    fig, axes = plt.subplots(6, 8, figsize=(20, 20))
    for ax, (sub, label) in zip(axes.flat, rows):
        brain = np.asarray(nib.load(args.out / sub / "seg.nii.gz").dataobj) > 0
        t1 = np.asarray(nib.load(args.out / sub / "t1_1mm.nii.gz").dataobj)
        edge = max(brain.sum(axis=(0, 2))[[0, -1]]) / brain.sum(axis=(0, 2)).max()
        site = scanner(sub)
        title = f"{sub}  {'PMG' if label else 'control'}  {site}  edge {edge:.2f}"
        panel(ax, t1, brain, title, SCANNER_COLOR[site])
    for ax in axes.flat[len(rows) :]:
        ax.set_axis_off()

    fig.suptitle(
        "task 5, widest axial slice at full field of view; red = brain against the FOV edge; "
        "title colour is the scanner (blue Skyra, orange Cigna)"
    )
    fig.tight_layout()
    fig.savefig(args.figures / "clipping.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
