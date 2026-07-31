"""Radar plot of the sweep: one filled ring per model, its radial thickness the bootstrap CI.

Spokes are grouped by dataset. Each task is scaled to its own [min ci_low, max ci_high] over
the plotted models, since scores are not comparable across tasks -- so the radial axis has no
shared units, and only the ordering and CI overlap within a spoke mean anything.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis_table import MODEL_NAMES, MODELS, OUT_DIR, TASK_NAMES, TASKS, load_runs

ROOT = Path(__file__).parent

# Inner radius of the donut. Each task's range maps to INNER..1, so the worst model on a task
# sits on the hole rather than at the centre, which would spike every ring through the middle.
INNER = 0.35

plt.style.use("seaborn-v0_8-talk")


def radar(runs: pd.DataFrame, path: Path) -> None:
    runs = runs[runs["model"].isin(MODELS) & runs["task"].isin(TASKS)]
    # Display names start with the dataset, so sorting on them groups by dataset then task.
    tasks = sorted(set(runs["task"]) & set(TASKS), key=lambda task: TASK_NAMES[task])

    scores = {
        key: runs.pivot(index="task", columns="model", values=key).reindex(tasks)[MODELS]
        for key in ("ci_low", "ci_high", "mean")
    }
    floor, ceiling = scores["ci_low"].min(axis=1), scores["ci_high"].max(axis=1)
    span = (ceiling - floor) / (1 - INNER)
    scaled = {
        key: frame.sub(floor - INNER * span, axis=0).div(span, axis=0)
        for key, frame in scores.items()
    }

    theta = np.linspace(0, 2 * np.pi, len(tasks), endpoint=False)
    wrap = lambda values: np.concatenate([values, values[:1]])  # noqa: E731
    angles = np.append(theta, theta[0] + 2 * np.pi)

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"projection": "polar"})
    for model, color in zip(MODELS, plt.rcParams["axes.prop_cycle"].by_key()["color"]):
        ax.fill_between(
            angles,
            wrap(scaled["ci_low"][model].to_numpy()),
            wrap(scaled["ci_high"][model].to_numpy()),
            color=color,
            alpha=0.35,
            label=MODEL_NAMES[model],
        )
        ax.plot(angles, wrap(scaled["mean"][model].to_numpy()), color=color)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(theta)
    ax.set_xticklabels([TASK_NAMES[task] for task in tasks])
    ax.set_ylim(0, 1)
    ax.set_yticklabels([])
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.05), frameon=False)

    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"wrote {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--path", type=Path, default=ROOT / "figures/radar.png")
    args = parser.parse_args()

    radar(load_runs(args.out_dir), args.path)
