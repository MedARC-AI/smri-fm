"""What a run claims, at the cut it is actually scored at.

Three rows per subject: the flair, the probability field, then hit / false / missed. Columns are
consecutive acquired slices centred on the largest tumour cross-section. The in-plane field is
uncropped, so a false positive anywhere in the head is visible.

Unlike task 4's version this is not optimistic: the saved fold carries the whole probability
volume, so the global cut can be drawn without refitting anything.

    uv run python figure_predictions.py --run ckpt-ptfull_folds
    uv run python figure_predictions.py --subjects sub-02 sub-18 sub-15 sub-01 sub-19 sub-16
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

OUT_DIR = Path(__file__).parent
TASK_DIR = Path("/data/smri-datasets/fomo_eval/Task_2")
CATEGORY_COLOURS = {"hit": (0.15, 0.9, 0.25), "false": (1.0, 0.25, 0.2), "missed": (0.3, 0.6, 1.0)}
N_SLICES = 5


def binarize(probability: np.ndarray, threshold: float) -> np.ndarray:
    """`Task2Method.binarize` at `largest_component=True`, so the panels match the score."""
    mask = probability >= threshold
    if not mask.any():
        return mask
    blobs, _ = ndimage.label(mask)
    sizes = np.bincount(blobs.reshape(-1))
    sizes[0] = 0
    return blobs == sizes.argmax()


def pick_subjects(curves) -> list[int]:
    """Best two, middle two and worst two by Dice at the global cut, as task 4 picks them."""
    best = int(curves["dice"].mean(0).argmax())
    order = np.argsort(-curves["dice"][:, best])
    middle = len(order) // 2
    return [order[0], order[1], order[middle - 1], order[middle], order[-2], order[-1]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="ckpt-ptfull_folds")
    parser.add_argument("--subjects", nargs="*", help="defaults to best two, middle two, worst two")
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()

    curves = np.load(OUT_DIR / f"output/{args.run}/curves.npz")
    names = [str(s) for s in curves["subjects"]]
    best = int(curves["dice"].mean(0).argmax())
    threshold = float(curves["thresholds"][best])
    picks = [names.index(s) for s in args.subjects] if args.subjects else pick_subjects(curves)

    fig, axes = plt.subplots(3 * len(picks), N_SLICES, figsize=(2.1 * N_SLICES, 6.6 * len(picks)))

    for row, index in enumerate(picks):
        subject = names[index]
        saved = np.load(OUT_DIR / f"output/{args.run}/folds/{subject}/prediction.npz")
        probability = saved["probability"]
        flair = nib.load(TASK_DIR / f"preprocessed/{subject}/ses-01/flair.nii.gz")
        data = np.asarray(flair.dataobj, dtype=np.float32)
        assert np.allclose(saved["affine"], flair.affine), f"{subject}: prediction affine differs"

        truth = np.zeros(probability.size, dtype=bool)
        truth[saved["truth_voxels"]] = True
        truth = truth.reshape(probability.shape)
        claimed = binarize(probability, threshold)

        centre = int(truth.sum(axis=(0, 1)).argmax())
        slices = np.clip(centre + np.arange(N_SLICES) - N_SLICES // 2, 0, truth.shape[2] - 1)
        dice = curves["dice"][index, best]
        print(
            f"{subject}: dice={dice:.3f} true={truth.sum()} claimed={claimed.sum()} "
            f"peak_prob={probability.max():.3f}"
        )

        for column, k in enumerate(slices):
            plane = np.rot90(data[:, :, k])
            limits = dict(vmin=0, vmax=np.percentile(data, 99.5))
            for offset in range(3):
                ax = axes[3 * row + offset][column]
                ax.imshow(plane, cmap="gray", **limits)
                ax.set_xticks([])
                ax.set_yticks([])

            axes[3 * row + 1][column].imshow(
                np.rot90(probability[:, :, k]), cmap="inferno", alpha=0.65, vmin=0, vmax=1
            )
            overlay = np.zeros((*plane.shape, 4), dtype=np.float32)
            hit, claim = np.rot90(truth[:, :, k]), np.rot90(claimed[:, :, k])
            for category, selection in (
                ("hit", hit & claim),
                ("false", ~hit & claim),
                ("missed", hit & ~claim),
            ):
                overlay[selection] = (*CATEGORY_COLOURS[category], 0.95)
            axes[3 * row + 2][column].imshow(overlay)
            axes[3 * row][column].set_title(f"slice {k}", fontsize=7)

        axes[3 * row][0].set_ylabel(f"{subject}\nDice {dice:.3f}  ({truth.sum()} vox)", fontsize=8)
        axes[3 * row + 1][0].set_ylabel(f"probability\npeak {probability.max():.2f}", fontsize=8)
        axes[3 * row + 2][0].set_ylabel(
            f"claim at {threshold:.3f}\n{claimed.sum()} vox", fontsize=8
        )

    fig.tight_layout()
    fig.legend(
        handles=[Patch(color=colour, label=name) for name, colour in CATEGORY_COLOURS.items()],
        loc="lower right",
        bbox_to_anchor=(1.0, 1.0),
        ncol=3,
        fontsize=9,
    )
    fig.suptitle(f"{args.run}: predictions at the global cut {threshold:.3f}", fontsize=11, y=1.003)
    path = OUT_DIR / f"figures/{args.run}_predictions{args.suffix}.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
