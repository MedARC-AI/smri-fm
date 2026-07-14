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

ADNI_EVAL_REPO_ID = "medarc/adni-mini"
IMAGE_COLUMN = "nifti"


def load_adni_eval() -> Dataset:
    return load_dataset(ADNI_EVAL_REPO_ID)["test"]


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
        metric_fns=(r2,),
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
    """Binary AD-vs-CN diagnosis classification (MCI dropped)."""
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
    """3-way diagnosis classification over CN / MCI / AD scans."""
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
def adni_amyloid_status(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    """Amyloid-PET positivity."""
    return ColumnTask(
        name="adni_amyloid_status",
        kind="classification",
        data=load_adni_eval(),
        image_column=IMAGE_COLUMN,
        target_column="amyloid_status",
        n_splits=n_splits,
        seed=seed,
        metric_fns=(bacc, auroc),
        positive_label=1.0,
    )


@register_task
def adni_amyloid_centiloid(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    """Amyloid burden in centiloids."""
    return ColumnTask(
        name="adni_amyloid_centiloid",
        kind="regression",
        data=load_adni_eval(),
        image_column=IMAGE_COLUMN,
        target_column="amyloid_centiloid",
        n_splits=n_splits,
        seed=seed,
        metric_fns=(pearson_r,),
    )


@register_task
def adni_tau_status(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    """Tau-PET positivity."""
    return ColumnTask(
        name="adni_tau_status",
        kind="classification",
        data=load_adni_eval(),
        image_column=IMAGE_COLUMN,
        target_column="tau_status",
        n_splits=n_splits,
        seed=seed,
        metric_fns=(bacc, auroc),
        positive_label=1.0,
    )


@register_task
def adni_tau_suvr(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    """Tau burden in meta-temporal SUVR."""
    return ColumnTask(
        name="adni_tau_suvr",
        kind="regression",
        data=load_adni_eval(),
        image_column=IMAGE_COLUMN,
        target_column="tau_suvr",
        n_splits=n_splits,
        seed=seed,
        metric_fns=(pearson_r,),
    )


@register_task
def adni_csf_abeta(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    """CSF Abeta42."""
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
    """CSF p-tau."""
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
    """CSF t-tau."""
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
def adni_mci_conversion(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    """MCI to AD conversion within 36 months."""
    return ColumnTask(
        name="adni_mci_conversion",
        kind="classification",
        data=load_adni_eval(),
        image_column=IMAGE_COLUMN,
        target_column="conversion_3y",
        n_splits=n_splits,
        seed=seed,
        metric_fns=(bacc, auroc),
        positive_label=1.0,
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
