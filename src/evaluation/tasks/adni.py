from datasets import Dataset, load_dataset

from evaluation.tasks.brain_age_gap import BrainAgeGapTask
from evaluation.tasks.column import ColumnTask
from evaluation.tasks.metrics import (
    auroc,
    bacc,
    pearson_r,
    r2,
    spearman_r,
)
from evaluation.tasks.registry import register_task

ADNI_EVAL_REPO_ID = "medarc/adni-mini-v1-1"
IMAGE_COLUMN = "nifti"


def load_adni_eval() -> Dataset:
    return load_dataset(ADNI_EVAL_REPO_ID)["eval"]


def _filter_diagnoses(data: Dataset, labels: set[str]) -> Dataset:
    names = data.features["diagnosis"].names
    keep = {names.index(label) for label in labels}
    return data.filter(lambda dx: dx in keep, input_columns="diagnosis")


@register_task
def adni_age(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    return ColumnTask(
        name="adni_age",
        kind="regression",
        data=load_adni_eval(),
        image_column=IMAGE_COLUMN,
        target_column="age",
        n_splits=n_splits,
        seed=seed,
        metric_fns=(r2, pearson_r),
    )


@register_task
def adni_sex(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    data = load_adni_eval()
    return ColumnTask(
        name="adni_sex",
        kind="classification",
        data=data,
        image_column=IMAGE_COLUMN,
        target_column="sex",
        n_splits=n_splits,
        seed=seed,
        metric_fns=(bacc, auroc),
        positive_label=data.features["sex"].names.index("Male"),
    )


@register_task
def adni_ad_cn(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    """Binary AD-vs-CN diagnosis classification (MCI dropped). Sanity task."""
    data = _filter_diagnoses(load_adni_eval(), {"CN", "AD"})
    return ColumnTask(
        name="adni_ad_cn",
        kind="classification",
        data=data,
        image_column=IMAGE_COLUMN,
        target_column="diagnosis",
        n_splits=n_splits,
        seed=seed,
        metric_fns=(bacc, auroc),
        positive_label=data.features["diagnosis"].names.index("AD"),
    )


@register_task
def adni_cn_mci_ad(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    """3-way diagnosis classification over CN / MCI / AD (staging)."""
    data = load_adni_eval()
    return ColumnTask(
        name="adni_cn_mci_ad",
        kind="classification",
        data=data,
        image_column=IMAGE_COLUMN,
        target_column="diagnosis",
        n_splits=n_splits,
        seed=seed,
        metric_fns=(bacc, auroc),
        positive_label=data.features["diagnosis"].names.index("AD"),
    )


@register_task
def adni_amyloid_centiloid(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    """Amyloid burden in centiloids (all diagnoses)."""
    return ColumnTask(
        name="adni_amyloid_centiloid",
        kind="regression",
        data=load_adni_eval(),
        image_column=IMAGE_COLUMN,
        target_column="amyloid_centiloid",
        n_splits=n_splits,
        seed=seed,
        metric_fns=(pearson_r, r2),
    )


@register_task
def adni_amyloid_centiloid_cn(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    """Centiloid within CN — primary preclinical amyloid task."""
    data = _filter_diagnoses(load_adni_eval(), {"CN"})
    return ColumnTask(
        name="adni_amyloid_centiloid_cn",
        kind="regression",
        data=data,
        image_column=IMAGE_COLUMN,
        target_column="amyloid_centiloid",
        n_splits=n_splits,
        seed=seed,
        metric_fns=(pearson_r, r2),
    )


@register_task
def adni_amyloid_centiloid_mci(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    data = _filter_diagnoses(load_adni_eval(), {"MCI"})
    return ColumnTask(
        name="adni_amyloid_centiloid_mci",
        kind="regression",
        data=data,
        image_column=IMAGE_COLUMN,
        target_column="amyloid_centiloid",
        n_splits=n_splits,
        seed=seed,
        metric_fns=(pearson_r, r2),
    )


@register_task
def adni_amyloid_centiloid_ad(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    data = _filter_diagnoses(load_adni_eval(), {"AD"})
    return ColumnTask(
        name="adni_amyloid_centiloid_ad",
        kind="regression",
        data=data,
        image_column=IMAGE_COLUMN,
        target_column="amyloid_centiloid",
        n_splits=n_splits,
        seed=seed,
        metric_fns=(pearson_r, r2),
    )


@register_task
def adni_tau_suvr(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    """Tau burden in meta-temporal SUVR (all diagnoses)."""
    return ColumnTask(
        name="adni_tau_suvr",
        kind="regression",
        data=load_adni_eval(),
        image_column=IMAGE_COLUMN,
        target_column="tau_suvr",
        n_splits=n_splits,
        seed=seed,
        metric_fns=(pearson_r, r2),
    )


@register_task
def adni_tau_suvr_cn(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    """Tau SUVR within CN — primary stratified tau task."""
    data = _filter_diagnoses(load_adni_eval(), {"CN}")
    return ColumnTask(
        name="adni_tau_suvr_cn",
        kind="regression",
        data=data,
        image_column=IMAGE_COLUMN,
        target_column="tau_suvr",
        n_splits=n_splits,
        seed=seed,
        metric_fns=(pearson_r, r2),
    )


@register_task
def adni_tau_suvr_mci(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    data = _filter_diagnoses(load_adni_eval(), {"MCI"})
    return ColumnTask(
        name="adni_tau_suvr_mci",
        kind="regression",
        data=data,
        image_column=IMAGE_COLUMN,
        target_column="tau_suvr",
        n_splits=n_splits,
        seed=seed,
        metric_fns=(pearson_r, r2),
    )


@register_task
def adni_csf_abeta(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    return ColumnTask(
        name="adni_csf_abeta",
        kind="regression",
        data=load_adni_eval(),
        image_column=IMAGE_COLUMN,
        target_column="csf_abeta",
        n_splits=n_splits,
        seed=seed,
        metric_fns=(spearman_r,),
    )


@register_task
def adni_csf_ptau(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    return ColumnTask(
        name="adni_csf_ptau",
        kind="regression",
        data=load_adni_eval(),
        image_column=IMAGE_COLUMN,
        target_column="csf_ptau",
        n_splits=n_splits,
        seed=seed,
        metric_fns=(spearman_r,),
    )


@register_task
def adni_csf_ttau(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    return ColumnTask(
        name="adni_csf_ttau",
        kind="regression",
        data=load_adni_eval(),
        image_column=IMAGE_COLUMN,
        target_column="csf_ttau",
        n_splits=n_splits,
        seed=seed,
        metric_fns=(spearman_r,),
    )


@register_task
def adni_csf_abeta_cn(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    data = _filter_diagnoses(load_adni_eval(), {"CN"})
    return ColumnTask(
        name="adni_csf_abeta_cn",
        kind="regression",
        data=data,
        image_column=IMAGE_COLUMN,
        target_column="csf_abeta",
        n_splits=n_splits,
        seed=seed,
        metric_fns=(spearman_r,),
    )


@register_task
def adni_mci_conversion_time(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    """Continuous time-to-AD (months) among labeled MCI subjects."""
    data = _filter_diagnoses(load_adni_eval(), {"MCI"})
    return ColumnTask(
        name="adni_mci_conversion_time",
        kind="regression",
        data=data,
        image_column=IMAGE_COLUMN,
        target_column="conversion_time_months",
        n_splits=n_splits,
        seed=seed,
        metric_fns=(pearson_r, r2),
    )


@register_task
def adni_synthseg_volumes(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    return ColumnTask(
        name="adni_synthseg_volumes",
        kind="regression",
        data=load_adni_eval(),
        image_column=IMAGE_COLUMN,
        target_column="synthseg_volumes",
        n_splits=n_splits,
        seed=seed,
        metric_fns=(r2,),
    )


@register_task
def adni_ad_cn_bag(seed: int = 0) -> ColumnTask:
    dataset = load_adni_eval()
    diagnosis_names = dataset.features["diagnosis"].names
    return BrainAgeGapTask(
        name="adni_ad_cn_bag",
        data=dataset,
        age_column="age",
        dx_column="diagnosis",
        control_label=diagnosis_names.index("CN"),
        case_label=diagnosis_names.index("AD"),
        image_column=IMAGE_COLUMN,
        seed=seed,
    )
