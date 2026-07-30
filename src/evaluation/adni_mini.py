"""Loader for the pinned SynthSeg-masked ADNI-mini evaluation snapshot."""

import os
from pathlib import Path

from datasets import Dataset, load_dataset


def load_adni_mini_eval() -> Dataset:
    snapshot = Path(os.environ["ADNI_MINI_SNAPSHOT"])
    return load_dataset(
        "parquet",
        data_files={"eval": str(snapshot / "data" / "*.parquet")},
        split="eval",
    )
