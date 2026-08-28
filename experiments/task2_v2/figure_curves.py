"""Threshold behaviour of a task2_v2 run, off `curves.npz` alone.

uv run python figure_curves.py     # -> figures/curves.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).parent
RUNS = ("ckpt-ptfull", "ckpt-ptfull_folds", "ckpt-walnut", "ckpt-walnut_folds")
BASELINE = OUT_DIR.parent / "fomo_tune_baseline/output/task2/curves.npz"
MAIN = "ckpt-ptfull_folds"


def load(path):
    d = np.load(path)
    best = int(d["dice"].mean(0).argmax())
    return d, best


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

    ax = axes[0][0]
    for run in RUNS:
        d, best = load(OUT_DIR / f"output/{run}/curves.npz")
        mean = d["dice"].mean(0)
        (line,) = ax.plot(d["thresholds"], mean, label=f"{run} ({mean[best]:.3f})")
        ax.plot(d["thresholds"][best], mean[best], "o", color=line.get_color())
    base, base_best = load(BASELINE)
    ax.plot(
        base["thresholds"],
        base["dice"].mean(0),
        "k--",
        label=f"baseline head ({base['dice'].mean(0)[base_best]:.3f})",
    )
    ax.set_xscale("log")
    ax.set_xlabel("threshold")
    ax.set_ylabel("mean Dice")
    ax.set_title("the selected cut sits on a noisy plateau")
    ax.legend(fontsize=7)

    d, best = load(OUT_DIR / f"output/{MAIN}/curves.npz")
    subjects, dice, thresholds = d["subjects"], d["dice"], d["thresholds"]
    oracle = dice.max(1)
    order = np.argsort(-oracle)

    ax = axes[0][1]
    y = np.arange(len(order))
    ax.barh(y, oracle[order], color="lightgrey", label="oracle cut")
    ax.barh(y, dice[order, best], height=0.5, label=f"global cut {thresholds[best]:.3f}")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{subjects[i]} ({d['true_voxels'][i]})" for i in order], fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("Dice")
    ax.set_title(f"{MAIN}: per subject, worst {(dice[:, best] == 0).sum()}/23 score zero")
    ax.legend(fontsize=7)

    ax = axes[1][0]
    ax.scatter(d["true_voxels"], d["predicted_voxels"][:, best], c="tab:blue")
    for i, subject in enumerate(subjects):
        ax.annotate(
            str(subject)[-2:], (d["true_voxels"][i], d["predicted_voxels"][i, best] + 1), fontsize=6
        )
    limits = [30, 3e5]
    ax.plot(limits, limits, "k--", lw=0.8, label="claimed = true")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(limits)
    ax.set_xlabel("true voxels")
    ax.set_ylabel("claimed voxels at the global cut")
    ax.set_title("claim size against truth, 1466x spread")
    ax.legend(fontsize=7)

    ax = axes[1][1]
    base_dice = base["dice"][:, base_best]
    ax.scatter(base_dice, dice[:, best])
    for i, subject in enumerate(subjects):
        ax.annotate(str(subject)[-2:], (base_dice[i], dice[i, best] + 0.015), fontsize=6)
    ax.plot([0, 0.85], [0, 0.85], "k--", lw=0.8)
    ax.set_xlabel("baseline head, Dice at its global cut")
    ax.set_ylabel(f"{MAIN}, Dice at its global cut")
    ax.set_title(f"paired, same checkpoint: {dice[:, best].mean() - base_dice.mean():+.3f} mean")

    fig.tight_layout()
    path = OUT_DIR / "figures/curves.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
