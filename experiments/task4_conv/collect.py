"""The four runs as one table, paired against the ridge head at the same config.

The seed pairs are the same config at two seeds, so their spread is the noise floor any
comparison in the first table has to clear.

    uv run python collect.py
"""

import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

OUT_DIR = Path(__file__).parent / "output"
RIDGE_DIR = OUT_DIR.parent.parent / "task4_logistic/output"

# the ridge run at the same scale, subcell and depth, one per checkpoint
BASELINES = {"pt-full": RIDGE_DIR / "ridge_ptfull", "walnut": RIDGE_DIR / "ridge_walnut"}
SEED_PAIRS = (("ckpt-ptfull", "ckpt-ptfull_s2"), ("ckpt-walnut", "ckpt-walnut_s2"))
CKPT_NAMES = {"pretrain_full_90_10_h100": "pt-full", "sub-52k": "walnut"}


def at_global(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Every subject's label-mean Dice at the run's own selected pair of cuts, and the subjects."""
    curves = np.load(run_dir / "curves.npz")
    dice = curves["dice"]
    mean = dice.mean(axis=(0, 1))
    best = np.unravel_index(mean.argmax(), mean.shape)
    return dice[:, :, best[0], best[1]].mean(axis=1), curves["subjects"]


def paired_delta(dice: np.ndarray, reference: np.ndarray, seed: int = 0) -> tuple[float, ...]:
    delta = dice - reference
    rng = np.random.default_rng(seed)
    samples = delta[rng.integers(0, len(delta), (2000, len(delta)))].mean(1)
    return delta.mean(), *np.percentile(samples, [2.5, 97.5])


def main() -> None:
    baselines = {name: at_global(path) for name, path in BASELINES.items()}

    print("| run | ckpt | seed | Dice | 95% CI | nerve | vessel | oracle | cuts | vs ridge | min |")
    print("|---" * 11 + "|")
    for run in sorted(d.name for d in OUT_DIR.iterdir() if (d / "metrics.json").exists()):
        metrics = json.loads((OUT_DIR / run / "metrics.json").read_text())
        cfg = OmegaConf.load(OUT_DIR / run / "config.yaml")
        ckpt = next((v for k, v in CKPT_NAMES.items() if k in cfg.ckpt_path), cfg.ckpt_path)
        dice, subjects = at_global(OUT_DIR / run)

        reference, reference_subjects = baselines[ckpt]
        assert (subjects == reference_subjects).all(), f"{run} is not on the baseline's subjects"
        mean, low, high = paired_delta(dice, reference)
        cuts = "/".join(f"{cut:.1e}" for cut in metrics["thresholds"])
        print(
            f"| {run} | {ckpt} | {cfg.seed} | **{metrics['dice']:.3f}** "
            f"| [{metrics['dice_ci_low']:.3f}, {metrics['dice_ci_high']:.3f}] "
            f"| {metrics['dice_nerve']:.3f} | {metrics['dice_vessel']:.3f} "
            f"| {metrics['dice_oracle']:.3f} | {cuts} "
            f"| {mean:+.3f} [{low:+.3f}, {high:+.3f}] "
            f"| {metrics['run_time'] / 60:.0f} |"
        )

    for name, (dice, _) in baselines.items():
        print(f"\nridge head, {name}: {dice.mean():.3f}")

    print("\n| seed pair | mean Dice | selected cuts | per-subject sd |")
    print("|---" * 4 + "|")
    for first, second in SEED_PAIRS:
        if not all((OUT_DIR / r / "metrics.json").exists() for r in (first, second)):
            continue
        a, _ = at_global(OUT_DIR / first)
        b, _ = at_global(OUT_DIR / second)
        cuts = [
            json.loads((OUT_DIR / r / "metrics.json").read_text())["thresholds"]
            for r in (first, second)
        ]
        shown = " / ".join("-".join(f"{cut:.1e}" for cut in pair) for pair in cuts)
        print(
            f"| {first} / {second} | {a.mean():.3f} / {b.mean():.3f} | {shown} | {np.std(a - b):.3f} |"
        )


if __name__ == "__main__":
    main()
