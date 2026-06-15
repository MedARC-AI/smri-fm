from __future__ import annotations

import os

from datasets import Dataset, load_dataset

from eval.datasets.registry import register_dataset
from eval.utils import Task, map_normalize

# Prediction target -> task type and head output dimension.
ADNI_TARGET_MAP = {
    "age": {"type": "regression", "output_dim": 1},
    "sex": {"type": "classification", "output_dim": 2},
    "diagnosis": {"type": "classification", "output_dim": 2},
    "synthseg_volumes": {"type": "regression", "output_dim": 101},
}

def _create_adni(target: str, repo_id: str, img_size, map_workers: int,
                 cache_dir: str | None = None, token: str | None = None,
                 **_kwargs) -> tuple[dict[str, Dataset], Task]:
    """Load ADNI, preprocess each split (cached, shared across targets), select target.

    Returns (splits, task): `splits` maps split -> a torch-formatted HF dataset with
    image/mask/target columns; `task` carries the type/output_dim the mains need.
    """
    if target not in ADNI_TARGET_MAP:
        raise ValueError(f"unknown ADNI target {target!r}; choices: {list(ADNI_TARGET_MAP)}")
    dataset = load_dataset(repo_id, os.getenv("HF_TOKEN"), cache_dir=cache_dir)

    splits = {}
    for split, split_ds in dataset.items():
        # Normalize once (target-agnostic, cached/shared), then pick the target.
        mapped = map_normalize(split_ds, tuple(img_size), map_workers)
        mapped = mapped.rename_column(target, "target")
        splits[split] = mapped.with_format(
            "torch", columns=["image", "mask", "target"], output_all_columns=True,
        )

    spec = ADNI_TARGET_MAP[target]
    return splits, Task(type=spec["type"], output_dim=spec["output_dim"], target="target")


@register_dataset
def adni_age(**kwargs):
    """ADNI age regression."""
    return _create_adni("age", **kwargs)


@register_dataset
def adni_sex(**kwargs):
    """ADNI sex classification (Female=0, Male=1)."""
    return _create_adni("sex", **kwargs)


@register_dataset
def adni_ad_cn(**kwargs):
    """ADNI AD vs CN classification."""
    return _create_adni("diagnosis", **kwargs)


@register_dataset
def adni_synthseg_volumes(**kwargs):
    """ADNI SynthSeg regional volume regression (101 outputs)."""
    return _create_adni("synthseg_volumes", **kwargs)
