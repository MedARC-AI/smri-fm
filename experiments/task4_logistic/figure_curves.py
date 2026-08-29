"""Where each subject and label sits, read entirely off `curves.npz`. No volumes, no refit.

Four panels: each label's Dice against its own cut with the other label held at the global cut,
the per-subject scores at the global cut with each subject's oracle marked, and claimed against
true voxels at the global cut, which is where over-claiming shows up.

    uv run python figure_curves.py --run logistic_walnut_1e1
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).parent
RUNS_DIR = OUT_DIR / "output"
LABEL_NAMES = ("nerve", "vessel")
LABEL_COLOURS = ("tab:red", "tab:blue")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="logistic_walnut_1e1")
    args = parser.parse_args()

    curves = np.load(RUNS_DIR / f"{args.run}/curves.npz")
    dice = curves["dice"]  # (subject, label, nerve cut, vessel cut)
    subjects = curves["subjects"]
    thresholds = curves["thresholds"]
    n_subjects, n_labels = dice.shape[:2]

    # the scored cuts: one pair for the cohort, maximizing the mean over subjects and labels
    shared_cut = np.unravel_index(dice.mean(axis=(0, 1)).argmax(), dice.shape[2:])
    shared = dice[:, :, shared_cut[0], shared_cut[1]]

    # each subject's own best pair, which is what the saved folds are drawn at
    by_subject = dice.mean(axis=1)
    oracle_cut = [np.unravel_index(subject.argmax(), subject.shape) for subject in by_subject]
    oracle = np.array([dice[s, :, i, j] for s, (i, j) in enumerate(oracle_cut)])

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    for label, (name, colour) in enumerate(zip(LABEL_NAMES, LABEL_COLOURS)):
        ax = axes[0][label]
        # the other label's cut held at its global value: the two compete for shared voxels
        other = shared_cut[1 - label]
        sweep = dice[:, label, :, other] if label == 0 else dice[:, label, other, :]
        ax.plot(thresholds, sweep.T, color=colour, alpha=0.25, linewidth=0.8)
        ax.plot(thresholds, sweep.mean(axis=0), color="black", linewidth=2, label="mean")
        ax.axvline(thresholds[shared_cut[label]], color="black", linestyle=":", label="global cut")
        ax.set_xscale("log")
        ax.set_xlabel(f"{name} cut")
        ax.set_ylabel("Dice")
        ax.set_title(f"{name}: every subject against its own cut")
        ax.legend(fontsize=8)

    ax = axes[1][0]
    order = np.argsort(-shared.mean(axis=1))
    x = np.arange(n_subjects)
    for label, (name, colour) in enumerate(zip(LABEL_NAMES, LABEL_COLOURS)):
        offset = 0.2 * (2 * label - 1)
        ax.bar(x + offset, shared[order, label], width=0.4, color=colour, label=f"{name} @ global")
        ax.plot(x + offset, oracle[order, label], "k.", markersize=3)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("sub-", "") for s in subjects[order]], fontsize=6)
    ax.set_xlabel("subject, best mean first")
    ax.set_ylabel("Dice")
    ax.set_title("per subject at the global cut; dots are that subject's oracle")
    ax.legend(fontsize=8)

    ax = axes[1][1]
    claimed = curves["predicted_voxels"][:, :, shared_cut[0], shared_cut[1]]
    true = curves["true_voxels"]
    for label, (name, colour) in enumerate(zip(LABEL_NAMES, LABEL_COLOURS)):
        ax.scatter(true[:, label], claimed[:, label], s=14, color=colour, label=name)
    limits = [0.5, max(true.max(), claimed.max()) * 1.5]
    ax.plot(limits, limits, "k:", linewidth=1, label="claimed = true")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*limits)
    ax.set_ylim(*limits)
    ax.set_xlabel("true voxels")
    ax.set_ylabel("claimed voxels at the global cut")
    ax.set_title("over- and under-claiming")
    ax.legend(fontsize=8)

    cuts = " / ".join(f"{thresholds[c]:.1e}" for c in shared_cut)
    fig.suptitle(
        f"{args.run}: mean {shared.mean():.3f} at cuts {cuts}, oracle {oracle.mean():.3f} "
        f"({n_subjects} subjects, {n_labels} labels)"
    )
    fig.tight_layout()
    (OUT_DIR / "figures").mkdir(exist_ok=True)
    path = OUT_DIR / f"figures/{args.run}_curves.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
