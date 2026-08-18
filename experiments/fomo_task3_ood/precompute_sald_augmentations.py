"""Warm the SALD augmentation cache in parallel, so GPU runs skip the ~10s/subject CPU work.

srun --partition=main --qos=high --account=sophont --cpus-per-task=32 --mem=64G --time=00:15:00 \
  uv run python experiments/fomo_task3_ood/precompute_sald_augmentations.py
"""

from concurrent.futures import ProcessPoolExecutor

from fomo_tune.datasets import load_fomo_task3
from fomo_tune.main_task3 import AUGMENT_CACHE, Config, cached_augment_row


def drain(row: dict) -> None:
    for _ in cached_augment_row(row, Config().seed):
        pass


if __name__ == "__main__":
    rows = list(load_fomo_task3())
    with ProcessPoolExecutor(max_workers=32) as pool:
        list(pool.map(drain, rows))
    print(f"warmed {len(rows) * 6} augmented views under {AUGMENT_CACHE}")
