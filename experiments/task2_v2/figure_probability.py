"""What the saved probabilities say that `curves.npz` cannot.

Four reads, all off `folds/<subject>/prediction.npz`:

- the component rule, swept: as scored (largest), off, and the best component an oracle could pick
- per-subject average precision, which ranks voxels without choosing a cut
- probability mass against true volume, i.e. whether the model knows how big the tumour is
- peak probability against Dice, i.e. whether the model knows when it has found nothing

    uv run python figure_probability.py     # -> figures/probability.png
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from sklearn.metrics import average_precision_score

OUT_DIR = Path(__file__).parent


def dice(claim: np.ndarray, truth: np.ndarray) -> float:
    denominator = claim.sum() + truth.sum()
    return 2 * np.logical_and(claim, truth).sum() / denominator if denominator else 1.0


def component_curves(probability, truth, thresholds):
    """Dice at every cut under no filter, the largest component, and the best component."""
    none_, largest, best = [], [], []
    for threshold in thresholds:
        mask = probability >= threshold
        none_.append(dice(mask, truth))
        if not mask.any():
            largest.append(dice(mask, truth))
            best.append(dice(mask, truth))
            continue
        blobs, count = ndimage.label(mask)
        sizes = np.bincount(blobs.reshape(-1))
        sizes[0] = 0
        largest.append(dice(blobs == sizes.argmax(), truth))
        # the tumour is one blob, so an oracle picking among them bounds any selection rule
        overlap = np.bincount(blobs[truth], minlength=count + 1)
        overlap[0] = 0
        best.append(dice(blobs == overlap.argmax(), truth) if overlap.any() else 0.0)
    return np.array(none_), np.array(largest), np.array(best)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="ckpt-ptfull_folds")
    args = parser.parse_args()

    curves = np.load(OUT_DIR / f"output/{args.run}/curves.npz")
    subjects = [str(s) for s in curves["subjects"]]
    thresholds, true_voxels = curves["thresholds"], curves["true_voxels"]
    best = int(curves["dice"].mean(0).argmax())
    scored = curves["dice"][:, best]

    rules, precision, mass, peak = [], [], [], []
    for subject in subjects:
        saved = np.load(OUT_DIR / f"output/{args.run}/folds/{subject}/prediction.npz")
        probability = saved["probability"]
        truth = np.zeros(probability.size, dtype=bool)
        truth[saved["truth_voxels"]] = True
        truth = truth.reshape(probability.shape)

        rules.append(component_curves(probability, truth, thresholds))
        precision.append(average_precision_score(truth.reshape(-1), probability.reshape(-1)))
        mass.append(float(probability.sum()))
        peak.append(float(probability.max()))
        print(f"{subject}: ap={precision[-1]:.3f} mass={mass[-1]:.0f} peak={peak[-1]:.3f}")

    rules = np.stack(rules)
    precision, mass, peak = np.array(precision), np.array(mass), np.array(peak)
    prevalence = true_voxels / np.array(
        [
            np.load(OUT_DIR / f"output/{args.run}/folds/{s}/prediction.npz")["probability"].size
            for s in subjects
        ]
    )

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

    ax = axes[0][0]
    for j, name in enumerate(("no filter", "largest component", "best component (oracle)")):
        mean = rules[:, j].mean(0)
        (line,) = ax.plot(thresholds, mean, label=f"{name} ({mean.max():.3f})")
        ax.plot(thresholds[mean.argmax()], mean.max(), "o", color=line.get_color())
    ax.set_xscale("log")
    ax.set_xlabel("threshold")
    ax.set_ylabel("mean Dice")
    ax.set_title("the largest component is often the wrong one")
    ax.legend(fontsize=7)

    ax = axes[0][1]
    ax.scatter(precision, scored)
    for i, subject in enumerate(subjects):
        ax.annotate(subject[-2:], (precision[i], scored[i] + 0.015), fontsize=6)
    ax.plot([0, 0.85], [0, 0.85], "k--", lw=0.8, label="Dice = AP")
    ax.set_xlabel("average precision (no cut chosen)")
    ax.set_ylabel("Dice at the global cut")
    ax.set_title("subjects with ranking but no Dice sit above the axis")
    ax.legend(fontsize=7)

    ax = axes[1][0]
    ax.scatter(true_voxels, mass)
    for i, subject in enumerate(subjects):
        ax.annotate(subject[-2:], (true_voxels[i], mass[i] * 1.08), fontsize=6)
    limits = [30, 3e5]
    ax.plot(limits, limits, "k--", lw=0.8, label="mass = true volume")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("true voxels")
    ax.set_ylabel("probability mass, sum of p over the volume")
    correlation = np.corrcoef(np.log(true_voxels), np.log(mass))[0, 1]
    ax.set_title(f"mass as a volume predictor, log-log r={correlation:.2f}")
    ax.legend(fontsize=7)

    ax = axes[1][1]
    found = scored > 0
    ax.scatter(peak[found], scored[found], label="Dice > 0")
    ax.scatter(peak[~found], scored[~found], marker="x", label="Dice = 0")
    for i, subject in enumerate(subjects):
        ax.annotate(subject[-2:], (peak[i], scored[i] + 0.015), fontsize=6)
    ax.axvline(thresholds[best], color="k", ls="--", lw=0.8, label=f"cut {thresholds[best]:.3f}")
    ax.set_xlabel("peak probability in the volume")
    ax.set_ylabel("Dice at the global cut")
    ax.set_title("peak probability does not separate found from missed")
    ax.legend(fontsize=7)

    fig.suptitle(f"{args.run}: what the probabilities say", fontsize=11)
    fig.tight_layout()
    path = OUT_DIR / f"figures/{args.run}_probability.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"\nwrote {path}")
    print(f"prevalence median {np.median(prevalence):.2e}, AP median {np.median(precision):.3f}")
    for j, name in enumerate(("no filter", "largest component", "best component")):
        mean = rules[:, j].mean(0)
        print(f"{name:>18}: best mean Dice {mean.max():.3f} at {thresholds[mean.argmax()]:.3f}")


if __name__ == "__main__":
    main()
