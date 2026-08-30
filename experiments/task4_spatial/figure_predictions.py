"""What one run claims, one row per label so the two do not overlap.

Three rows a subject: the display box unannotated, then the nerve and the vessel as
hit / false / missed. Columns are slices centred on each side's nerve. Subjects are drawn in
tiers -- the best, the middle and the worst by the oracle mean, which is what the folds are cut at.

    uv run python figure_predictions.py --run train-2_test-4_alpha-1e1
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.patches import Patch
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
TASK_DIR = ROOT / "data/fomo_eval/Task_4"
OUT_DIR = Path(__file__).parent
RUNS_DIR = OUT_DIR / "output"

# the display box of `explore_fomo_task4/model_box.py`: 128 x 96 x 96 at 0.5mm, holding every
# subject's labels
BOX = np.array([128, 96, 96])
CENTRE = np.array([0.0, 4.0, -16.0])

LABEL_NAMES = ("nerve", "vessel")
CATEGORY_COLOURS = {"hit": (0.15, 0.9, 0.25), "false": (1.0, 0.25, 0.2), "missed": (0.3, 0.6, 1.0)}
CONNECTIVITY = np.ones((3, 3, 3))
N_SLICES = 7
SLICE_STRIDE = 2


def window(panel: np.ndarray) -> dict:
    lo, hi = np.percentile(panel, [1.0, 99.8])
    return {"vmin": lo, "vmax": max(hi, lo + 1)}


def nerve_sides(labels: np.ndarray) -> list[float]:
    """Each nerve component's axis-0 centre. Label 1 has exactly two components in every subject."""
    components, count = ndimage.label(labels == 1, structure=CONNECTIVITY)
    coords = {i: np.argwhere(components == i) for i in range(1, count + 1)}
    sides = sorted(coords, key=lambda i: -len(coords[i]))[:2]
    return sorted(coords[i][:, 0].mean() for i in sides)


def load_prediction(run: str, subject: str, seg: nib.Nifti1Image) -> np.ndarray:
    saved = np.load(RUNS_DIR / run / "folds" / subject / "prediction.npz")
    assert (saved["shape"] == seg.shape).all(), f"{subject}: prediction is not on the label grid"
    assert np.allclose(saved["affine"], seg.affine), f"{subject}: prediction affine differs"
    claimed = np.zeros(int(np.prod(saved["shape"])), dtype=np.uint8)
    claimed[saved["voxels"]] = saved["labels"]
    return claimed.reshape(saved["shape"])


def pick_subjects(run: str, per_tier: int) -> tuple[list[str], np.ndarray, np.ndarray]:
    """The best, middle and worst `per_tier` by the oracle mean, which is what the folds are cut at."""
    curves = np.load(RUNS_DIR / run / "curves.npz")
    dice, subjects = curves["dice"], curves["subjects"]

    global_cut = np.unravel_index(dice.mean(axis=(0, 1)).argmax(), dice.shape[2:])
    at_global = dice[:, :, global_cut[0], global_cut[1]]
    by_subject = dice.mean(axis=1)
    oracle = np.array(
        [
            dice[s, :, *np.unravel_index(subject.argmax(), subject.shape)]
            for s, subject in enumerate(by_subject)
        ]
    )

    order = np.argsort(-oracle.mean(axis=1))
    middle = len(order) // 2 - per_tier // 2
    picks = [*order[:per_tier], *order[middle : middle + per_tier], *order[-per_tier:]]
    return [str(subjects[i]) for i in picks], oracle[picks], at_global[picks]


def draw_panel(ax, plane: np.ndarray, limits: dict) -> None:
    ax.imshow(np.rot90(plane), cmap="gray", **limits)
    ax.set_xticks([])
    ax.set_yticks([])


def draw_categories(ax, plane, truth, claimed, limits) -> None:
    """One label painted hit / false / missed, filled rather than outlined: at 0.5mm the structures
    are a couple of voxels across and a contour of that is unreadable."""
    draw_panel(ax, plane, limits)
    overlay = np.zeros((*plane.shape, 4), dtype=np.float32)
    for category, mask in (
        ("hit", truth & claimed),
        ("false", ~truth & claimed),
        ("missed", truth & ~claimed),
    ):
        overlay[mask] = (*CATEGORY_COLOURS[category], 0.95)
    ax.imshow(np.rot90(overlay))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="train-2_test-4_alpha-1e1")
    parser.add_argument("--per-tier", type=int, default=3)
    args = parser.parse_args()

    subjects, oracle, at_global = pick_subjects(args.run, args.per_tier)
    n_columns = 2 * N_SLICES
    n_rows = 3 * len(subjects)
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(1.6 * n_columns, 1.7 * n_rows))

    for s, subject in enumerate(subjects):
        image = nib.load(TASK_DIR / f"preprocessed/{subject}/ses-01/t2w.nii.gz")
        seg = nib.load(TASK_DIR / f"labels/{subject}/ses-01/seg.nii.gz")
        data = np.asarray(image.dataobj, dtype=np.float32)
        labels = np.asarray(seg.dataobj).round().astype(np.uint8)
        claimed = load_prediction(args.run, subject, seg)

        anchor = np.array(ndimage.center_of_mass(data > data.mean()))
        lo = np.round(anchor + CENTRE - BOX / 2).astype(int)
        hi = lo + BOX
        assert (lo >= 0).all() and (hi <= data.shape).all(), f"{subject}: box leaves the volume"
        box = tuple(slice(a, b) for a, b in zip(lo, hi))

        # a claim outside the display box is invisible here, so say how much of it is inside
        shown = [
            (claimed[box] == value).sum() / max((claimed == value).sum(), 1) for value in (1, 2)
        ]

        columns = np.concatenate(
            [
                np.clip(
                    int(round(centre - lo[0]))
                    + SLICE_STRIDE * (np.arange(N_SLICES) - N_SLICES // 2),
                    0,
                    BOX[0] - 1,
                )
                for centre in nerve_sides(labels)
            ]
        )

        for c, k in enumerate(columns):
            plane = data[lo[0] + k, box[1], box[2]]
            truth_plane = labels[lo[0] + k, box[1], box[2]]
            claimed_plane = claimed[lo[0] + k, box[1], box[2]]
            limits = window(plane)

            draw_panel(axes[3 * s][c], plane, limits)
            for label, value in enumerate((1, 2)):
                draw_categories(
                    axes[3 * s + 1 + label][c],
                    plane,
                    truth_plane == value,
                    claimed_plane == value,
                    limits,
                )
            axes[3 * s][c].set_title(f"{'lr'[c // N_SLICES]} R+{k}", fontsize=6)

        axes[3 * s][0].set_ylabel(
            f"{subject}\noracle {oracle[s].mean():.3f} / global {at_global[s].mean():.3f}",
            fontsize=7,
        )
        for label, name in enumerate(LABEL_NAMES):
            axes[3 * s + 1 + label][0].set_ylabel(
                f"{name}\n{oracle[s][label]:.3f} / {at_global[s][label]:.3f}\n"
                f"{100 * shown[label]:.0f}% in box",
                fontsize=7,
            )

    fig.tight_layout()
    fig.legend(
        handles=[Patch(color=colour, label=name) for name, colour in CATEGORY_COLOURS.items()],
        loc="lower right",
        bbox_to_anchor=(1.0, 1.0),
        ncol=3,
        fontsize=9,
    )
    fig.suptitle(f"{args.run}: predictions at each subject's oracle cut", fontsize=11, y=1.004)
    (OUT_DIR / "figures").mkdir(exist_ok=True)
    path = OUT_DIR / f"figures/{args.run}_predictions.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
