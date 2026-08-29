"""Out-of-fold probability for every run: what cortex pooling does to individual subjects.

One row per checkpoint, one column per pooling setting, in the order `collect.py` prints. Only
the probabilities differ across a row, so a subject can be followed left to right. The dotted
line is the highest-scoring negative -- every case above it is separated.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXP_DIR = Path(__file__).parent
OUT_DIR = EXP_DIR / "output"
FIG_DIR = EXP_DIR / "figures"

CKPTS = (("ptfull", "pt-full"), ("walnut", "walnut-vitl"))
POOLS = (
    ("global", "global"),
    ("cortex000", "cortex, frac 0.0"),
    ("cortex010", "cortex, frac 0.1"),
    ("cortex025", "cortex, frac 0.25"),
)


def read_preds(name: str) -> list[dict]:
    return [json.loads(line) for line in (OUT_DIR / name / "preds.json").read_text().splitlines()]


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(len(CKPTS), len(POOLS), figsize=(17, 9), sharey=True)

    for row, (ckpt, ckpt_label) in enumerate(CKPTS):
        for col, (pool, pool_label) in enumerate(POOLS):
            name = f"ckpt-{ckpt}_pool-{pool}"
            rows = read_preds(name)
            metrics = json.loads((OUT_DIR / name / "metrics.json").read_text())
            y = np.array([r["label"] for r in rows])
            p = np.array([r["pred"] for r in rows])

            ax = axes[row, col]
            jitter = np.random.default_rng(0).uniform(-0.12, 0.12, len(y))
            for label in (0, 1):
                keep = y == label
                ax.scatter(label + jitter[keep], p[keep], s=24, color=f"C{label}")
            for r, x, z in zip(rows, y + jitter, p):
                ax.annotate(
                    r["subject"].removeprefix("sub_"),
                    (x, z),
                    fontsize=6,
                    xytext=(4, 0),
                    textcoords="offset points",
                )

            ax.axhline(p[y == 0].max(), ls=":", c="k", lw=0.8)
            ax.set_xticks([0, 1])
            ax.set_xlim(-0.4, 1.4)
            ax.set_xlabel("label", fontsize=8)
            ax.set_title(
                f"{pool_label}\nAUROC {metrics['auroc']:.3f} "
                f"({metrics['auroc_ci_low']:.3f}–{metrics['auroc_ci_high']:.3f})",
                fontsize=9,
            )
        axes[row, 0].set_ylabel(f"{ckpt_label}\n\nout-of-fold p", fontsize=9)

    fig.suptitle("Task 5 cortex pooling: out-of-fold scores, every run", y=1.001)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "scores.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
