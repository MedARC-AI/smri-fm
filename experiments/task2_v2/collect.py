"""The four runs as one table, plus what the twin pairs say about run-to-run noise.

The pairs are the same config at the same seeds, so their spread is the noise floor any
comparison in the first table has to clear.

    uv run python collect.py
"""

import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

OUT_DIR = Path(__file__).parent / "output"
BASELINE = OUT_DIR.parent.parent / "fomo_tune_baseline/output/task2/curves.npz"

# the twin of each run: same config and seeds, rerun once the folds were saved
TWINS = (("ckpt-ptfull", "ckpt-ptfull_folds"), ("ckpt-walnut", "ckpt-walnut_folds"))
CKPT_NAMES = {"pretrain_full_90_10_h100": "pt-full", "sub-52k": "walnut"}


def at_global(path: Path) -> np.ndarray:
    """Every subject's Dice at the run's own selected cut."""
    curves = np.load(path)
    return curves["dice"][:, int(curves["dice"].mean(0).argmax())]


def paired_delta(
    dice: np.ndarray, reference: np.ndarray, seed: int = 0
) -> tuple[float, float, float]:
    delta = dice - reference
    rng = np.random.default_rng(seed)
    samples = delta[rng.integers(0, len(delta), (2000, len(delta)))].mean(1)
    return delta.mean(), *np.percentile(samples, [2.5, 97.5])


def main() -> None:
    baseline = at_global(BASELINE)

    print("| run | ckpt | Dice | 95% CI | oracle | cut | zeros | paired vs baseline head | min |")
    print("|---" * 9 + "|")
    for run in sorted(d.name for d in OUT_DIR.iterdir() if (d / "metrics.json").exists()):
        metrics = json.loads((OUT_DIR / run / "metrics.json").read_text())
        cfg = OmegaConf.load(OUT_DIR / run / "config.yaml")
        ckpt = next((v for k, v in CKPT_NAMES.items() if k in cfg.ckpt_path), cfg.ckpt_path)
        dice = at_global(OUT_DIR / run / "curves.npz")
        mean, low, high = paired_delta(dice, baseline)
        print(
            f"| {run} | {ckpt} | **{metrics['dice']:.3f}** "
            f"| [{metrics['dice_ci_low']:.3f}, {metrics['dice_ci_high']:.3f}] "
            f"| {metrics['dice_oracle']:.3f} | {metrics['threshold']:.3f} "
            f"| {int((dice == 0).sum())}/{len(dice)} "
            f"| {mean:+.3f} [{low:+.3f}, {high:+.3f}] "
            f"| {metrics['run_time'] / 60:.0f} |"
        )

    print(
        f"\nbaseline head: {baseline.mean():.3f}, {int((baseline == 0).sum())}/{len(baseline)} zeros"
    )
    print("\n| twin pair | mean Dice | selected cut | per-subject sd |")
    print("|---" * 4 + "|")
    for first, second in TWINS:
        a, b = at_global(OUT_DIR / first / "curves.npz"), at_global(OUT_DIR / second / "curves.npz")
        cuts = [
            json.loads((OUT_DIR / r / "metrics.json").read_text())["threshold"]
            for r in (first, second)
        ]
        print(
            f"| {first} / {second} | {a.mean():.3f} / {b.mean():.3f} "
            f"| {cuts[0]:.4f} / {cuts[1]:.4f} | {np.std(a - b):.3f} |"
        )


if __name__ == "__main__":
    main()
