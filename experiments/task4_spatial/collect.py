"""The runs as one table, each paired against the un-augmented baseline.

Every run sees the same folds and the same protocol, so per-subject Dice are paired and the delta
is far tighter than the overlap of two marginal CIs. The two alpha controls are read the same way:
`base_a3` against `base` is what the effective penalty alone buys, and `train_tta` minus that is
what is left for the augmentation.

    uv run python collect.py
"""

import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

OUT_DIR = Path(__file__).parent / "output"
BASELINE = "train-0_test-0_alpha-1e1"


def at_global(run: str) -> np.ndarray:
    """Every subject's mean-over-labels Dice at the run's own selected pair of cuts."""
    dice = np.load(OUT_DIR / run / "curves.npz")["dice"]
    mean = dice.mean(axis=(0, 1))
    nerve, vessel = np.unravel_index(mean.argmax(), mean.shape)
    return dice[:, :, nerve, vessel].mean(axis=1)


def paired_delta(dice: np.ndarray, reference: np.ndarray, seed: int = 0) -> tuple[float, ...]:
    delta = dice - reference
    rng = np.random.default_rng(seed)
    samples = delta[rng.integers(0, len(delta), (2000, len(delta)))].mean(axis=1)
    return delta.mean(), *np.percentile(samples, [2.5, 97.5])


def main() -> None:
    runs = sorted(d.name for d in OUT_DIR.iterdir() if (d / "metrics.json").exists())
    assert BASELINE in runs, f"{BASELINE} has not finished; every delta is read against it"
    baseline = at_global(BASELINE)

    print(
        "| run | train views | tta views | alpha | Dice | 95% CI | nerve | vessel | oracle "
        "| nerve cut | vessel cut | paired vs base | min |"
    )
    print("|---" * 13 + "|")
    for run in runs:
        metrics = json.loads((OUT_DIR / run / "metrics.json").read_text())
        cfg = OmegaConf.load(OUT_DIR / run / "config.yaml")
        nerve_cut, vessel_cut = metrics["thresholds"]
        mean, low, high = paired_delta(at_global(run), baseline)
        train = cfg.train_views + 1 if cfg.train_spatial else 1
        tta = cfg.tta_views + 1 if cfg.tta_spatial else 1
        print(
            f"| {run} | {train} | {tta} | {cfg.alpha:.2g} "
            f"| **{metrics['dice']:.3f}** "
            f"| [{metrics['dice_ci_low']:.3f}, {metrics['dice_ci_high']:.3f}] "
            f"| {metrics['dice_nerve']:.3f} | {metrics['dice_vessel']:.3f} "
            f"| {metrics['dice_oracle']:.3f} | {nerve_cut:.1e} | {vessel_cut:.1e} "
            f"| {mean:+.3f} [{low:+.3f}, {high:+.3f}] "
            f"| {metrics['run_time'] / 60:.0f} |"
        )


if __name__ == "__main__":
    main()
