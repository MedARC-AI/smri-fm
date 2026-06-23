from functools import lru_cache
import json
import logging

from datasets import Dataset, load_from_disk
from huggingface_hub import snapshot_download

from evaluation.tasks.brain_age_gap import BrainAgeGapTask
from evaluation.tasks.column import ColumnTask
from evaluation.tasks.metrics import (
    age_bias_correct_predictions,
    auprc,
    auroc,
    balanced_accuracy,
    cohen_kappa,
    mae,
    multiclass_auroc,
    pearson_r,
    r2,
    spearman_r,
)
from evaluation.tasks.registry import register_task

logger = logging.getLogger(__name__)

ADNI_EVAL_REPO_ID = "medarc/adni_eval"
ADNI_EVAL_REVISION = "e81062568b00363ced2a552e156ddb7db471e204"
ADNI_EVAL_SOURCE = {
    "dataset_repo": ADNI_EVAL_REPO_ID,
    "dataset_revision": ADNI_EVAL_REVISION,
}
GROUP_COLUMN = "participant_id"
IMAGE_COLUMN = "nifti"
COVARIATES = ("age", "sex")


def load_adni_eval() -> Dataset:
    """Download and load the pinned ADNI evaluation dataset from the Hub."""
    logger.info(f"dataset_source: {json.dumps(ADNI_EVAL_SOURCE, sort_keys=True)}")
    return _load_adni_eval()


@lru_cache(maxsize=1)
def _load_adni_eval() -> Dataset:
    path = snapshot_download(
        ADNI_EVAL_REPO_ID,
        repo_type="dataset",
        revision=ADNI_EVAL_REVISION,
        allow_patterns=["dataset_dict.json", "eval/*"],
    )
    data = load_from_disk(path)["eval"]
    return data


def _filter_diagnoses(data: Dataset, labels: set[str]) -> Dataset:
    names = data.features["diagnosis"].names
    keep = {names.index(label) for label in labels}
    return data.filter(lambda dx: dx in keep, input_columns="diagnosis")

# --- v1 biomarker tasks (single-scan input; sparse labels auto-filtered) -----

@register_task
def adni_age(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    return ColumnTask(
        name="adni_age",
        kind="regression",
        data=load_adni_eval(),
        metrics=(pearson_r, r2, mae),
        target_column="age",
        image_column=IMAGE_COLUMN,
        group_column=GROUP_COLUMN,
        n_splits=n_splits,
        seed=seed,
        prediction_postprocessor=age_bias_correct_predictions,
        participant_level=True,
    )


@register_task
def adni_sex(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    data = load_adni_eval()
    return ColumnTask(
        name="adni_sex",
        kind="classification",
        data=data,
        metrics=(balanced_accuracy, auroc),
        target_column="sex",
        image_column=IMAGE_COLUMN,
        group_column=GROUP_COLUMN,
        n_splits=n_splits,
        seed=seed,
        positive_label=data.features["sex"].names.index("Male"),
        participant_level=True,
    )


@register_task
def adni_ad_cn(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    """Binary AD-vs-CN diagnosis classification (MCI dropped)."""
    data = _filter_diagnoses(load_adni_eval(), {"CN", "AD"})
    return ColumnTask(
        name="adni_ad_cn",
        kind="classification",
        data=data,
        metrics=(auroc, auprc, balanced_accuracy),
        target_column="diagnosis",
        image_column=IMAGE_COLUMN,
        group_column=GROUP_COLUMN,
        n_splits=n_splits,
        seed=seed,
        positive_label=data.features["diagnosis"].names.index("AD"),
        selection_metric="roc_auc",
        covariate_columns=COVARIATES,
        participant_level=True,
    )


@register_task
def adni_cn_mci_ad(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    """3-way diagnosis classification over all CN / MCI / AD scans."""
    return ColumnTask(
        name="adni_cn_mci_ad",
        kind="classification",
        data=load_adni_eval(),
        metrics=(multiclass_auroc, balanced_accuracy, cohen_kappa),
        target_column="diagnosis",
        image_column=IMAGE_COLUMN,
        group_column=GROUP_COLUMN,
        n_splits=n_splits,
        seed=seed,
        covariate_columns=COVARIATES,
        participant_level=True,
    )


@register_task
def adni_amyloid_status(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    """Amyloid-PET positivity (UC Berkeley AMYLOID_STATUS)."""
    return ColumnTask(
        name="adni_amyloid_status",
        kind="classification",
        data=load_adni_eval(),
        metrics=(auroc, auprc, balanced_accuracy),
        target_column="amyloid_status",
        image_column=IMAGE_COLUMN,
        group_column=GROUP_COLUMN,
        n_splits=n_splits,
        seed=seed,
        positive_label=1.0,
        selection_metric="roc_auc",
        covariate_columns=COVARIATES,
    )


@register_task
def adni_amyloid_centiloid(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    """Amyloid burden in Centiloids (UC Berkeley CENTILOIDS)."""
    return ColumnTask(
        name="adni_amyloid_centiloid",
        kind="regression",
        data=load_adni_eval(),
        metrics=(pearson_r, r2, mae),
        target_column="amyloid_centiloid",
        image_column=IMAGE_COLUMN,
        group_column=GROUP_COLUMN,
        n_splits=n_splits,
        seed=seed,
        covariate_columns=COVARIATES,
    )


@register_task
def adni_tau_status(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    """Tau-PET positivity (meta-temporal SUVR > 1.23, Jack 2017)."""
    return ColumnTask(
        name="adni_tau_status",
        kind="classification",
        data=load_adni_eval(),
        metrics=(auroc, auprc, balanced_accuracy),
        target_column="tau_status",
        image_column=IMAGE_COLUMN,
        group_column=GROUP_COLUMN,
        n_splits=n_splits,
        seed=seed,
        positive_label=1.0,
        selection_metric="roc_auc",
        covariate_columns=COVARIATES,
    )


@register_task
def adni_tau_suvr(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    """Tau burden (meta-temporal SUVR)."""
    return ColumnTask(
        name="adni_tau_suvr",
        kind="regression",
        data=load_adni_eval(),
        metrics=(pearson_r, r2, mae),
        target_column="tau_suvr",
        image_column=IMAGE_COLUMN,
        group_column=GROUP_COLUMN,
        n_splits=n_splits,
        seed=seed,
        covariate_columns=COVARIATES,
    )


@register_task
def adni_csf_abeta(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    """CSF Abeta42, pg/mL (Elecsys, censored values clipped to assay limits)."""
    return ColumnTask(
        name="adni_csf_abeta",
        kind="regression",
        data=load_adni_eval(),
        metrics=(spearman_r, r2),
        target_column="csf_abeta",
        image_column=IMAGE_COLUMN,
        group_column=GROUP_COLUMN,
        n_splits=n_splits,
        seed=seed,
        covariate_columns=COVARIATES,
    )


@register_task
def adni_csf_ptau(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    """CSF p-tau, pg/mL (Elecsys)."""
    return ColumnTask(
        name="adni_csf_ptau",
        kind="regression",
        data=load_adni_eval(),
        metrics=(spearman_r, r2),
        target_column="csf_ptau",
        image_column=IMAGE_COLUMN,
        group_column=GROUP_COLUMN,
        n_splits=n_splits,
        seed=seed,
        covariate_columns=COVARIATES,
    )


@register_task
def adni_csf_ttau(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    """CSF t-tau, pg/mL (Elecsys)."""
    return ColumnTask(
        name="adni_csf_ttau",
        kind="regression",
        data=load_adni_eval(),
        metrics=(spearman_r, r2),
        target_column="csf_ttau",
        image_column=IMAGE_COLUMN,
        group_column=GROUP_COLUMN,
        n_splits=n_splits,
        seed=seed,
        covariate_columns=COVARIATES,
    )


@register_task
def adni_mci_conversion(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    """Prognostic MCI -> AD conversion within 36 months (single baseline scan)."""
    return ColumnTask(
        name="adni_mci_conversion",
        kind="classification",
        data=load_adni_eval(),
        metrics=(auroc, auprc, balanced_accuracy),
        target_column="conversion_3y",
        image_column=IMAGE_COLUMN,
        group_column=GROUP_COLUMN,
        n_splits=n_splits,
        seed=seed,
        positive_label=1.0,
        selection_metric="roc_auc",
        covariate_columns=COVARIATES,
    )


@register_task
def adni_synthseg_volumes(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    return ColumnTask(
        name="adni_synthseg_volumes",
        kind="regression",
        data=load_adni_eval(),
        metrics=(r2, pearson_r),
        target_column="synthseg_volumes",
        image_column=IMAGE_COLUMN,
        group_column=GROUP_COLUMN,
        n_splits=n_splits,
        seed=seed,
        participant_level=True,
    )


@register_task
def adni_ad_cn_bag() -> BrainAgeGapTask:
    data = load_adni_eval()
    diagnosis_names = data.features["diagnosis"].names
    return BrainAgeGapTask(
        name="adni_ad_cn_bag",
        data=data,
        age_column="age",
        dx_column="diagnosis",
        control_label=diagnosis_names.index("CN"),
        case_label=diagnosis_names.index("AD"),
        image_column=IMAGE_COLUMN,
        group_column=GROUP_COLUMN,
    )
