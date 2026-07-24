import os
from functools import lru_cache

from datasets import Dataset, load_from_disk
from huggingface_hub import snapshot_download

from evaluation.tasks.brain_age_gap import BrainAgeGapTask
from evaluation.tasks.column import ColumnTask
from evaluation.tasks.metrics import (
    auprc,
    auroc,
    bacc,
    pearson_r,
    r2,
    spearman_r,
)
from evaluation.tasks.registry import register_task

PPMI_EVAL_REPO_ID = "medarc/ppmi-mini"
IMAGE_COLUMN = "nifti"

# Clinical targets are joined in at load time from a local parquet built by
# datasets/ppmi/build_slopes.py. They are NOT in the published dataset: the PPMI
# DUA forbids redistributing subject-level data to unregistered parties.
CLINICAL_PARQUET = os.environ.get(
    "PPMI_CLINICAL_PARQUET",
    "/mnt/data/medarc/datasets/ppmi/derived/ppmi_mini_clinical.parquet",
)


@lru_cache(maxsize=1)
def load_ppmi_eval() -> Dataset:
    """ppmi-mini is a save_to_disk arrow dataset with a single 'eval' split."""
    path = snapshot_download(
        PPMI_EVAL_REPO_ID,
        repo_type="dataset",
        allow_patterns=["dataset_dict.json", "eval/*"],
    )
    return load_from_disk(path)["eval"]


@lru_cache(maxsize=1)
def load_ppmi_clinical() -> Dataset:
    """ppmi-mini + clinical target columns, aligned on sample_id."""
    import pandas as pd

    data = load_ppmi_eval()
    clinical = pd.read_parquet(CLINICAL_PARQUET).set_index("sample_id")
    missing = set(data["sample_id"]) - set(clinical.index)
    if missing:
        raise ValueError(
            f"{len(missing)} samples absent from {CLINICAL_PARQUET}; rebuild it"
        )
    clinical = clinical.loc[list(data["sample_id"])]
    for col in clinical.columns:
        if col != "PATNO":
            data = data.add_column(col, clinical[col].tolist())
    return data


def _filter_diagnoses(data: Dataset, labels: set[str]) -> Dataset:
    names = data.features["diagnosis"].names
    keep = {names.index(label) for label in labels}
    return data.filter(lambda dx: dx in keep, input_columns="diagnosis")


# ---------------------------------------------------------------------------
# Basic / sanity
# ---------------------------------------------------------------------------


@register_task
def ppmi_age(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    return ColumnTask(
        name="ppmi_age",
        kind="regression",
        data=load_ppmi_eval(),
        image_column=IMAGE_COLUMN,
        target_column="age",
        n_splits=n_splits,
        seed=seed,
        metric_fns=(r2, pearson_r),
    )


@register_task
def ppmi_sex(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    """Sanity classification (expect near-saturated AUROC)."""
    data = load_ppmi_eval()
    return ColumnTask(
        name="ppmi_sex",
        kind="classification",
        data=data,
        image_column=IMAGE_COLUMN,
        target_column="sex",
        n_splits=n_splits,
        seed=seed,
        metric_fns=(bacc, auroc, auprc),
        positive_label=data.features["sex"].names.index("Male"),
    )


@register_task
def ppmi_pd_cn(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    """Binary PD-vs-control. SWEDD and Prodromal dropped (label noise)."""
    data = _filter_diagnoses(load_ppmi_eval(), {"CN", "PD"})
    return ColumnTask(
        name="ppmi_pd_cn",
        kind="classification",
        data=data,
        image_column=IMAGE_COLUMN,
        target_column="diagnosis",
        n_splits=n_splits,
        seed=seed,
        metric_fns=(bacc, auroc, auprc),
        positive_label=data.features["diagnosis"].names.index("PD"),
    )


@register_task
def ppmi_diagnosis(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    """4-way CN / PD / Prodromal / SWEDD."""
    data = load_ppmi_eval()
    return ColumnTask(
        name="ppmi_diagnosis",
        kind="classification",
        data=data,
        image_column=IMAGE_COLUMN,
        target_column="diagnosis",
        n_splits=n_splits,
        seed=seed,
        metric_fns=(bacc,),
        positive_label=data.features["diagnosis"].names.index("PD"),
    )


@register_task
def ppmi_pd_cn_bag() -> BrainAgeGapTask:
    """Brain-age gap association (PD cases vs CN-trained age residual)."""
    data = load_ppmi_eval()
    names = data.features["diagnosis"].names
    return BrainAgeGapTask(
        name="ppmi_pd_cn_bag",
        data=data,
        age_column="age",
        dx_column="diagnosis",
        control_label=names.index("CN"),
        case_label=names.index("PD"),
        image_column=IMAGE_COLUMN,
    )


# ---------------------------------------------------------------------------
# Prognosis: annualized slopes over 48 months from the scan
# ---------------------------------------------------------------------------


@register_task
def ppmi_updrs3_slope_48m(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    """Annualized MDS-UPDRS Part III motor progression. Higher = worsening.

    OFF-medication and untreated exams only; ON scores are drug-suppressed.
    """
    return ColumnTask(
        name="ppmi_updrs3_slope_48m",
        kind="regression",
        data=load_ppmi_clinical(),
        image_column=IMAGE_COLUMN,
        target_column="np3tot_slope_48m",
        n_splits=n_splits,
        seed=seed,
        metric_fns=(pearson_r, spearman_r, r2),
    )


@register_task
def ppmi_moca_slope_48m(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    """Annualized MoCA global-cognition slope. Lower = declining."""
    return ColumnTask(
        name="ppmi_moca_slope_48m",
        kind="regression",
        data=load_ppmi_clinical(),
        image_column=IMAGE_COLUMN,
        target_column="mcatot_slope_48m",
        n_splits=n_splits,
        seed=seed,
        metric_fns=(pearson_r, spearman_r, r2),
    )


@register_task
def ppmi_hoehn_yahr_slope_48m(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    """Annualized Hoehn & Yahr stage progression."""
    return ColumnTask(
        name="ppmi_hoehn_yahr_slope_48m",
        kind="regression",
        data=load_ppmi_clinical(),
        image_column=IMAGE_COLUMN,
        target_column="nhy_slope_48m",
        n_splits=n_splits,
        seed=seed,
        metric_fns=(pearson_r, spearman_r, r2),
    )


# ---------------------------------------------------------------------------
# Cross-sectional severity at the time of the scan (baseline controls for the
# slope tasks: if the model only reads current severity, these saturate first)
# ---------------------------------------------------------------------------


@register_task
def ppmi_updrs3_baseline(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    return ColumnTask(
        name="ppmi_updrs3_baseline",
        kind="regression",
        data=load_ppmi_clinical(),
        image_column=IMAGE_COLUMN,
        target_column="np3tot_baseline",
        n_splits=n_splits,
        seed=seed,
        metric_fns=(pearson_r, spearman_r, r2),
    )


@register_task
def ppmi_hoehn_yahr_baseline(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    """Ordinal 0-5 severity stage, treated as regression."""
    return ColumnTask(
        name="ppmi_hoehn_yahr_baseline",
        kind="regression",
        data=load_ppmi_clinical(),
        image_column=IMAGE_COLUMN,
        target_column="nhy_baseline",
        n_splits=n_splits,
        seed=seed,
        metric_fns=(pearson_r, spearman_r, r2),
    )


@register_task
def ppmi_moca_baseline(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    return ColumnTask(
        name="ppmi_moca_baseline",
        kind="regression",
        data=load_ppmi_clinical(),
        image_column=IMAGE_COLUMN,
        target_column="mcatot_baseline",
        n_splits=n_splits,
        seed=seed,
        metric_fns=(pearson_r, spearman_r, r2),
    )
