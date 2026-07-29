"""ADNI-mini eval tasks: predict a clinical column from frozen T1 features.

One HF dataset carries per-scan labels; each task picks a target column and drops scans
missing that label (many PET/CSF measures are absent).
"""

import numpy as np
from datasets import Dataset, load_dataset

from nanobrain.eval.tasks import register_task
from nanobrain.eval.tasks.base import ClassificationTask, RegressionTask

ADNI_REPO_ID = "medarc/adni-mini"
ADNI_FILES = f"hf://datasets/{ADNI_REPO_ID}/data/eval-*.parquet"
IMAGE_COL = "nifti"


def load_adni() -> Dataset:
    # TODO: revert back to the usual load_dataset approach once the dataset is fixed
    return load_dataset("parquet", data_files={"eval": ADNI_FILES}, split="eval")


def drop_missing(dataset: Dataset, target_col: str) -> Dataset:
    finite = np.isfinite(np.asarray(dataset[target_col], dtype=float))
    return dataset.select(np.flatnonzero(finite))


@register_task
def adni_age() -> RegressionTask:
    return RegressionTask(
        name="adni_age",
        dataset_fn=lambda: drop_missing(load_adni(), "age"),
        target_col="age",
        image_col=IMAGE_COL,
    )


@register_task
def adni_sex() -> ClassificationTask:
    # sex is a ClassLabel(['Female', 'Male']); its raw index gives Male=1 as the positive class.
    return ClassificationTask(
        name="adni_sex",
        dataset_fn=load_adni,
        target_col="sex",
        image_col=IMAGE_COL,
    )


@register_task
def adni_ad_cn() -> ClassificationTask:
    """AD vs CN diagnosis (MCI dropped)."""
    names = load_adni().features["diagnosis"].names
    cn, ad = names.index("CN"), names.index("AD")

    def dataset_fn() -> Dataset:
        return load_adni().filter(lambda dx: dx in (cn, ad), input_columns="diagnosis")

    return ClassificationTask(
        name="adni_ad_cn",
        dataset_fn=dataset_fn,
        target_col="diagnosis",
        image_col=IMAGE_COL,
        target_map={cn: 0, ad: 1},
    )


@register_task
def adni_amyloid_centiloid() -> RegressionTask:
    """Amyloid burden in centiloids."""
    return RegressionTask(
        name="adni_amyloid_centiloid",
        dataset_fn=lambda: drop_missing(load_adni(), "amyloid_centiloid"),
        target_col="amyloid_centiloid",
        image_col=IMAGE_COL,
    )


@register_task
def adni_tau_suvr() -> RegressionTask:
    """Tau burden in meta-temporal SUVR."""
    return RegressionTask(
        name="adni_tau_suvr",
        dataset_fn=lambda: drop_missing(load_adni(), "tau_suvr"),
        target_col="tau_suvr",
        image_col=IMAGE_COL,
    )
