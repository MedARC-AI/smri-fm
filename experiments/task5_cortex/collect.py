"""The cortex-pooling sweep as one table."""

import json
from pathlib import Path

from omegaconf import OmegaConf

EXP_DIR = Path(__file__).parent
OUT_DIR = EXP_DIR / "output"

CKPT_NAMES = {"pretrain_full_90_10_h100": "pt-full", "sub-52k": "walnut-vitl"}

ORDER = tuple(
    f"ckpt-{ckpt}_pool-{pool}"
    for ckpt in ("ptfull", "walnut")
    for pool in ("global", "cortex000", "cortex010", "cortex025")
)

HEADER = "| ckpt | pooling | cortex_frac | AUROC | 95% CI | time |"
RULE = "|---" * 6 + "|"


def main() -> None:
    names = [name for name in ORDER if (OUT_DIR / name / "metrics.json").exists()]

    print(HEADER)
    print(RULE)
    for name in names:
        cfg = OmegaConf.load(OUT_DIR / name / "config.yaml")
        m = json.loads((OUT_DIR / name / "metrics.json").read_text())
        frac = f"{cfg.cortex_frac:g}" if cfg.pooling == "cortex" else "--"

        print(
            f"| {CKPT_NAMES[Path(cfg.ckpt_path).parent.name]} | {cfg.pooling} | {frac} "
            f"| **{m['auroc']:.3f}** | {m['auroc_ci_low']:.3f} – {m['auroc_ci_high']:.3f} "
            f"| {m['run_time']:.0f}s |"
        )


if __name__ == "__main__":
    main()
