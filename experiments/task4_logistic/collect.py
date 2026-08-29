"""The runs as one table, each paired against the ridge/pt-full baseline.

Ridge and logistic at the same checkpoint see the same folds and the same feature cache, so their
per-subject Dice are paired and the delta is far tighter than the overlap of two marginal CIs.

    uv run python collect.py
"""

import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

OUT_DIR = Path(__file__).parent / "output"
BASELINE = "ridge_ptfull"

# the grid `s4_c4_d04` was scored on, before the floor was truncated to 1e-3
OLD_GRID_BASELINE = 0.252

CKPT_NAMES = {"pretrain_full_90_10_h100": "pt-full", "sub-52k": "walnut"}


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
        "| run | head | ckpt | alpha | Dice | 95% CI | nerve | vessel | oracle "
        "| nerve cut | vessel cut | paired vs ridge/pt-full | min |"
    )
    print("|---" * 13 + "|")
    for run in runs:
        metrics = json.loads((OUT_DIR / run / "metrics.json").read_text())
        cfg = OmegaConf.load(OUT_DIR / run / "config.yaml")
        ckpt = next((v for k, v in CKPT_NAMES.items() if k in cfg.ckpt_path), cfg.ckpt_path)
        nerve_cut, vessel_cut = metrics["thresholds"]
        mean, low, high = paired_delta(at_global(run), baseline)
        alpha = f"{cfg.alpha:.0e}" if cfg.head == "logistic" else "-"
        print(
            f"| {run} | {cfg.head} | {ckpt} | {alpha} "
            f"| **{metrics['dice']:.3f}** "
            f"| [{metrics['dice_ci_low']:.3f}, {metrics['dice_ci_high']:.3f}] "
            f"| {metrics['dice_nerve']:.3f} | {metrics['dice_vessel']:.3f} "
            f"| {metrics['dice_oracle']:.3f} | {nerve_cut:.1e} | {vessel_cut:.1e} "
            f"| {mean:+.3f} [{low:+.3f}, {high:+.3f}] "
            f"| {metrics['run_time'] / 60:.0f} |"
        )

    print(
        f"\nridge/pt-full is {baseline.mean():.3f} here against {OLD_GRID_BASELINE:.3f} for the same"
        f" config as `s4_c4_d04`;\nthe difference is the truncated threshold grid, not the method."
    )


if __name__ == "__main__":
    main()
