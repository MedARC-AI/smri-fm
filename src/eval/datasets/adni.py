from __future__ import annotations

import os

import torch
from datasets import Dataset, DatasetDict, load_dataset

from eval.datasets.registry import register_dataset

DEFAULT_REPO_ID = "medarc/adni_eval"

# Target map: prediction target -> task kind and head output dimension.
ADNI_TARGET_MAP = {
    "age": {"kind": "regression", "output_dim": 1},
    "sex": {"kind": "classification", "output_dim": 2},
    "diagnosis": {"kind": "classification", "output_dim": 2},
    "synthseg_volumes": {"kind": "regression", "output_dim": 101},
}


class ADNIDataset(torch.utils.data.Dataset):
    """ADNI scans wrapped for a single prediction target.

    The task kind (classification vs regression) and head output dimension are
    looked up from ADNI_TARGET_MAP, so one class covers every ADNI target.
    """

    def __init__(self, dataset: Dataset, target: str):
        if target not in ADNI_TARGET_MAP:
            raise ValueError(f"unknown ADNI target {target!r}; choices: {list(ADNI_TARGET_MAP)}")
        self.target = target
        self.kind = ADNI_TARGET_MAP[target]["kind"]
        self.output_dim = ADNI_TARGET_MAP[target]["output_dim"]
        self.dataset = dataset

    def with_images(self, dataset: Dataset) -> "ADNIDataset":
        """Swap in a transformed HF dataset (with image/mask columns) in place."""
        self.dataset = dataset.with_format(
            "torch", columns=["image", "mask", self.target], output_all_columns=True,
        )
        return self

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        target = sample[self.target]
        sample["target"] = target.long() if self.kind == "classification" else target.float()
        return sample


def select_training_size(dataset: DatasetDict, size: int = 2000) -> DatasetDict:
    if size not in {500, 1000, 2000}:
        raise ValueError("training size must be one of 500, 1000, or 2000")
    train = dataset["train"].filter(lambda rank: rank < size, input_columns="train_rank")
    return DatasetDict(train=train, validation=dataset["validation"], test=dataset["test"])


def _create_adni(target: str, repo_id: str = DEFAULT_REPO_ID,
                 cache_dir: str | None = None, **_kwargs) -> dict[str, ADNIDataset]:
    dataset = load_dataset(repo_id, token=os.getenv("HF_TOKEN"), cache_dir=cache_dir)
    return {split: ADNIDataset(split_ds, target) for split, split_ds in dataset.items()}


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
