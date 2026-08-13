"""Re-run the task 3 baseline protocol, keeping the embeddings and per-subject predictions.

`main_task3.py` reports only pooled metrics, so the baseline run dir has no way to look at an
individual subject. This calls the same `Task3Method` and the same `cross_validate`, so the
metrics it prints must match `experiments/fomo_tune_baseline/output/task3/metrics.json` -- that
equality is the check that these embeddings are the ones the score was made of.

Needs a GPU. ~6 minutes.
"""

from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from fomo_tune.datasets import load_fomo_task3
from fomo_tune.main_task3 import Config, Task3Method, cross_validate, score
from fomo_tune.utils import set_seed

OUT_DIR = Path(__file__).parent
BASELINE = OUT_DIR.parent / "fomo_tune_baseline/output/task3/config.yaml"


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.structured(Config), OmegaConf.load(BASELINE))
    set_seed(cfg.seed)
    print(OmegaConf.to_yaml(cfg).rstrip(), flush=True)

    rows = list(load_fomo_task3())
    method = Task3Method(cfg)
    y, oof = cross_validate(rows, method)

    # every subject is in the training side of at least one fold, so the cache is complete
    subjects = [row["subject"] for row in rows]
    embeds = np.stack([method.cache[subject] for subject in subjects])

    np.savez(OUT_DIR / "oof.npz", subjects=np.array(subjects), age=y, pred=oof, embed=embeds)

    # the embeddings are gitignored with every other .npz, so the predictions also go out as text
    lines = ["subject\tage\tpred"]
    lines += [f"{s}\t{a:.0f}\t{p:.3f}" for s, a, p in zip(subjects, y, oof)]
    (OUT_DIR / "oof.tsv").write_text("\n".join(lines) + "\n")

    print({k: round(v, 4) for k, v in score(y, oof).items()}, flush=True)


if __name__ == "__main__":
    main()
