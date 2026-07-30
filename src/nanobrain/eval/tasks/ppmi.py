"""PPMI-mini eval tasks: predict a clinical column from frozen T1 features.

One HF dataset carries per-scan labels for 1,000 subjects, one T1w scan each: age, sex and a
four-way diagnosis (CN, PD, Prodromal, SWEDD).

There is no site or scanner column, so scan date is the only handle on the acquisition, and it
is a strong one: the cohorts were enrolled in waves, so scan year alone reads PD vs CN at 0.61
AUROC and Prodromal vs CN at 0.84. Header geometry (data shape plus voxel size) is a second,
independent scanner proxy, worth 0.79 on the raw Prodromal-vs-CN split. Diagnosis tasks
therefore pair each case with a control from the same (scan-year band, age band, sex) cell,
which drops scan year, age, sex and geometry back to 0.43-0.52.
"""

import numpy as np
from datasets import Dataset, load_dataset

from nanobrain.eval.tasks import register_task
from nanobrain.eval.tasks.base import ClassificationTask, RegressionTask
from nanobrain.eval.tasks.utils import matched_indices

PPMI_REPO_ID = "medarc/ppmi-mini"
PPMI_REVISION = "c8453a0e039ad07cd43fe82832d0e49b871f02cd"
# In the path, not load_dataset(revision=...), which the packaged arrow builder ignores.
PPMI_FILES = f"hf://datasets/{PPMI_REPO_ID}@{PPMI_REVISION}/eval/data-*.arrow"
IMAGE_COL = "nifti"
# The only non-3D volume in the release: a (512, 512, 78, 2) two-phase acquisition, which the
# models' 3D preprocessing cannot consume.
EXCLUDED_SAMPLES = ("sub-3200_ses-20101202_T1w",)
AGE_BINS = (55, 65, 72)
YEAR_BINS = (2013, 2017, 2021)
MATCH_SEED = 0


def load_ppmi() -> Dataset:
    # TODO: revert back to the usual load_dataset approach once the dataset is fixed
    dataset = load_dataset("arrow", data_files={"eval": PPMI_FILES}, split="eval")
    return dataset.filter(
        lambda sample_id: sample_id not in EXCLUDED_SAMPLES, input_columns="sample_id"
    )


def diagnosis_indices(*names: str) -> tuple[int, ...]:
    """ClassLabel indices for the named diagnoses, which the raw `diagnosis` column holds."""
    levels = load_ppmi().features["diagnosis"].names
    return tuple(levels.index(name) for name in names)


def matched_cohort(dataset: Dataset, negative: int, positive: int) -> Dataset:
    """The two-class subcohort in which scan-year band, age band and sex carry no label signal."""
    diagnosis = np.asarray(dataset["diagnosis"])
    subset = dataset.select(np.flatnonzero(np.isin(diagnosis, [negative, positive])))

    labels = (np.asarray(subset["diagnosis"]) == positive).astype(int).tolist()
    years = [int(date[:4]) for date in subset["scan_date"]]
    cells = list(
        zip(
            np.digitize(years, YEAR_BINS),
            np.digitize(subset["age"], AGE_BINS),
            subset["sex"],
        )
    )
    return subset.select(matched_indices(cells, labels, seed=MATCH_SEED))


@register_task
def ppmi_age() -> RegressionTask:
    """Age over the whole cohort, 999 subjects spanning 31-86.

    Unmatched: age is what the disease cohorts differ in, and the acquisition proxies barely
    predict it on their own (geometry r = 0.05, scan year r = 0.27).
    """
    return RegressionTask(
        name="ppmi_age",
        dataset_fn=load_ppmi,
        target_col="age",
        image_col=IMAGE_COL,
    )


@register_task
def ppmi_pd_cn() -> ClassificationTask:
    """PD vs CN, matched: 426 subjects, 213 per class."""
    cn, pd_dx = diagnosis_indices("CN", "PD")
    return ClassificationTask(
        name="ppmi_pd_cn",
        dataset_fn=lambda: matched_cohort(load_ppmi(), cn, pd_dx),
        target_col="diagnosis",
        image_col=IMAGE_COL,
        target_map={cn: 0, pd_dx: 1},
    )


@register_task
def ppmi_pd_prodromal() -> ClassificationTask:
    """Manifest PD vs prodromal (at-risk, not yet diagnosed), matched: 324 subjects, 162 per class.

    The hardest split in the release -- prodromal subjects are enrolled on hyposmia or REM-sleep
    behaviour disorder, so any structural difference is early -- but the confound floors are at
    chance, so anything above it is signal.
    """
    prodromal, pd_dx = diagnosis_indices("Prodromal", "PD")
    return ClassificationTask(
        name="ppmi_pd_prodromal",
        dataset_fn=lambda: matched_cohort(load_ppmi(), prodromal, pd_dx),
        target_col="diagnosis",
        image_col=IMAGE_COL,
        target_map={prodromal: 0, pd_dx: 1},
    )
