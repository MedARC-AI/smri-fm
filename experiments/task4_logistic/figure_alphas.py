"""What the sweep says, read off `metrics.json` and `curves.npz`. No volumes, no refit.

Four panels: Dice against alpha with ridge as the reference line, the per-subject paired delta
that the comparison actually rests on, the two heads' threshold sweeps on one axis, and claimed
against true voxels.

    uv run python figure_alphas.py     # -> figures/alphas.png
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

OUT_DIR = Path(__file__).parent
RUNS_DIR = OUT_DIR / "output"

CKPTS = {"pt-full": "tab:blue", "walnut": "tab:orange"}
CKPT_OF = {"pretrain_full_90_10_h100": "pt-full", "sub-52k": "walnut"}
BEST, REFERENCE = "logistic_walnut_1e1", "ridge_walnut"
LABEL_NAMES = ("nerve", "vessel")
LABEL_COLOURS = ("tab:red", "tab:blue")


def load(run: str) -> tuple[dict, dict, np.ndarray]:
    cfg = OmegaConf.load(RUNS_DIR / run / "config.yaml")
    metrics = json.loads((RUNS_DIR / run / "metrics.json").read_text())
    return cfg, metrics, np.load(RUNS_DIR / run / "curves.npz")


def at_global(curves: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """Every subject's mean-over-labels Dice at the run's own selected pair of cuts."""
    cut = np.unravel_index(curves.mean(axis=(0, 1)).argmax(), curves.shape[2:])
    return curves[:, :, cut[0], cut[1]].mean(axis=1), cut


def main() -> None:
    runs = {}
    for run_dir in sorted(RUNS_DIR.iterdir()):
        if not (run_dir / "metrics.json").exists():
            continue
        cfg, metrics, curves = load(run_dir.name)
        ckpt = next(v for k, v in CKPT_OF.items() if k in cfg.ckpt_path)
        runs[run_dir.name] = (cfg, metrics, curves, ckpt)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0][0]
    for ckpt, colour in CKPTS.items():
        points = sorted(
            (cfg.alpha, m["dice"], m["dice_ci_low"], m["dice_ci_high"])
            for cfg, m, _, c in runs.values()
            if cfg.head == "logistic" and c == ckpt
        )
        alpha, dice, low, high = (np.array(v) for v in zip(*points))
        ax.errorbar(
            alpha,
            dice,
            yerr=[dice - low, high - dice],
            marker="o",
            color=colour,
            capsize=3,
            label=f"logistic, {ckpt}",
        )
        ridge = next(
            m["dice"] for cfg, m, _, c in runs.values() if cfg.head == "ridge" and c == ckpt
        )
        ax.axhline(ridge, color=colour, linestyle=":", label=f"ridge, {ckpt}")
    ax.set_xscale("log")
    ax.set_xlabel("alpha")
    ax.set_ylabel("Dice")
    ax.set_title("logistic against alpha; ridge is the dotted reference")
    ax.legend(fontsize=8)

    ax = axes[0][1]
    best, _ = at_global(runs[BEST][2]["dice"])
    reference, _ = at_global(runs[REFERENCE][2]["dice"])
    delta = best - reference
    order = np.argsort(-delta)
    subjects = runs[BEST][2]["subjects"]
    colours = ["tab:green" if d >= 0 else "tab:red" for d in delta[order]]
    ax.bar(np.arange(len(delta)), delta[order], color=colours)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(delta.mean(), color="black", linestyle="--", label=f"mean {delta.mean():+.3f}")
    ax.set_xticks(np.arange(len(delta)))
    ax.set_xticklabels([s.replace("sub-", "") for s in subjects[order]], fontsize=6)
    ax.set_xlabel("subject")
    ax.set_ylabel("Dice difference")
    ax.set_title(f"paired: {BEST} - {REFERENCE}, {int((delta > 0).sum())}/{len(delta)} better")
    ax.legend(fontsize=8)

    ax = axes[1][0]
    for run, style in [(REFERENCE, "-"), (BEST, "-"), ("logistic_walnut_1e4", "--")]:
        _, _, curves, _ = runs[run]
        dice, cut = at_global(curves["dice"])
        sweep = curves["dice"].mean(axis=(0, 1))[:, cut[1]]
        ax.plot(curves["thresholds"], sweep, style, label=f"{run} (peak {sweep.max():.3f})")
    ax.set_xscale("log")
    ax.set_xlabel("nerve cut, vessel held at its own best")
    ax.set_ylabel("mean Dice")
    ax.set_title("the score scale moves with the head")
    ax.legend(fontsize=7)

    ax = axes[1][1]
    for run, marker in [(REFERENCE, "s"), (BEST, "o")]:
        _, _, curves, _ = runs[run]
        _, cut = at_global(curves["dice"])
        claimed = curves["predicted_voxels"][:, :, cut[0], cut[1]]
        true = curves["true_voxels"]
        for label, (name, colour) in enumerate(zip(LABEL_NAMES, LABEL_COLOURS)):
            ax.scatter(
                true[:, label],
                claimed[:, label],
                s=16,
                marker=marker,
                color=colour,
                alpha=0.6,
                label=f"{name}, {run.split('_')[0]}",
            )
    limits = [50, 1e4]
    ax.plot(limits, limits, "k:", linewidth=1, label="claimed = true")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("true voxels")
    ax.set_ylabel("claimed voxels at the global cut")
    ax.set_title("over- and under-claiming")
    ax.legend(fontsize=7)

    fig.suptitle(
        f"task4_logistic: best {BEST} {runs[BEST][1]['dice']:.3f} "
        f"against {REFERENCE} {runs[REFERENCE][1]['dice']:.3f}"
    )
    fig.tight_layout()
    (OUT_DIR / "figures").mkdir(exist_ok=True)
    path = OUT_DIR / "figures/alphas.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
