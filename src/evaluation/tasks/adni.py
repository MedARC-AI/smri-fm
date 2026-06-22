from functools import lru_cache

from datasets import Dataset, load_from_disk
from huggingface_hub import snapshot_download

from evaluation.tasks.brain_age_gap import BrainAgeGapTask
from evaluation.tasks.column import ColumnTask
from evaluation.tasks.registry import register_task

REPO_ID = "medarc/adni_eval"
GROUP_COLUMN = "participant_id"
IMAGE_COLUMN = "nifti"


@lru_cache(maxsize=1)
def load_adni_eval() -> Dataset:
    """Download and load the serialized v1 `eval` split from the Hub."""
    path = snapshot_download(
        REPO_ID,
        repo_type="dataset",
        allow_patterns=["dataset_dict.json", "eval/*"],
    )
    dataset = load_from_disk(path)["eval"]
    return dataset


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
        target_column="age",
        image_column=IMAGE_COLUMN,
        group_column=GROUP_COLUMN,
        n_splits=n_splits,
        seed=seed,
    )


@register_task
def adni_sex(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    return ColumnTask(
        name="adni_sex",
        kind="classification",
        data=load_adni_eval(),
        target_column="sex",
        image_column=IMAGE_COLUMN,
        group_column=GROUP_COLUMN,
        n_splits=n_splits,
        seed=seed,
    )


@register_task
def adni_ad_cn(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    """Binary AD-vs-CN diagnosis classification (MCI dropped)."""
    data = _filter_diagnoses(load_adni_eval(), {"CN", "AD"})
    return ColumnTask(
        name="adni_ad_cn",
        kind="classification",
        data=data,
        target_column="diagnosis",
        image_column=IMAGE_COLUMN,
        group_column=GROUP_COLUMN,
        n_splits=n_splits,
        seed=seed,
    )


@register_task
def adni_cn_mci_ad(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    """3-way diagnosis classification over all CN / MCI / AD scans."""
    return ColumnTask(
        name="adni_cn_mci_ad",
        kind="classification",
        data=load_adni_eval(),
        target_column="diagnosis",
        image_column=IMAGE_COLUMN,
        group_column=GROUP_COLUMN,
        n_splits=n_splits,
        seed=seed,
    )


# --- v1 biomarker tasks (single-scan input; sparse labels auto-filtered) -----

@register_task
def adni_amyloid_status(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    """Amyloid-PET positivity (UC Berkeley AMYLOID_STATUS)."""
    return ColumnTask(
        name="adni_amyloid_status",
        kind="classification",
        data=load_adni_eval(),
        target_column="amyloid_status",
        image_column=IMAGE_COLUMN,
        group_column=GROUP_COLUMN,
        n_splits=n_splits,
        seed=seed,
    )


@register_task
def adni_amyloid_centiloid(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    """Amyloid burden in Centiloids (UC Berkeley CENTILOIDS)."""
    return ColumnTask(
        name="adni_amyloid_centiloid",
        kind="regression",
        data=load_adni_eval(),
        target_column="amyloid_centiloid",
        image_column=IMAGE_COLUMN,
        group_column=GROUP_COLUMN,
        n_splits=n_splits,
        seed=seed,
    )


@register_task
def adni_tau_status(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    """Tau-PET positivity (meta-temporal SUVR > 1.23, Jack 2017)."""
    return ColumnTask(
        name="adni_tau_status",
        kind="classification",
        data=load_adni_eval(),
        target_column="tau_status",
        image_column=IMAGE_COLUMN,
        group_column=GROUP_COLUMN,
        n_splits=n_splits,
        seed=seed,
    )


@register_task
def adni_tau_suvr(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    """Tau burden (meta-temporal SUVR)."""
    return ColumnTask(
        name="adni_tau_suvr",
        kind="regression",
        data=load_adni_eval(),
        target_column="tau_suvr",
        image_column=IMAGE_COLUMN,
        group_column=GROUP_COLUMN,
        n_splits=n_splits,
        seed=seed,
    )


@register_task
def adni_csf_abeta(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    """CSF Abeta42, pg/mL (Elecsys, censored values clipped to assay limits)."""
    return ColumnTask(
        name="adni_csf_abeta",
        kind="regression",
        data=load_adni_eval(),
        target_column="csf_abeta",
        image_column=IMAGE_COLUMN,
        group_column=GROUP_COLUMN,
        n_splits=n_splits,
        seed=seed,
    )


@register_task
def adni_csf_ptau(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    """CSF p-tau, pg/mL (Elecsys)."""
    return ColumnTask(
        name="adni_csf_ptau",
        kind="regression",
        data=load_adni_eval(),
        target_column="csf_ptau",
        image_column=IMAGE_COLUMN,
        group_column=GROUP_COLUMN,
        n_splits=n_splits,
        seed=seed,
    )


@register_task
def adni_csf_ttau(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    """CSF t-tau, pg/mL (Elecsys)."""
    return ColumnTask(
        name="adni_csf_ttau",
        kind="regression",
        data=load_adni_eval(),
        target_column="csf_ttau",
        image_column=IMAGE_COLUMN,
        group_column=GROUP_COLUMN,
        n_splits=n_splits,
        seed=seed,
    )


@register_task
def adni_mci_conversion(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    """Prognostic MCI -> AD conversion within 36 months (single baseline scan)."""
    return ColumnTask(
        name="adni_mci_conversion",
        kind="classification",
        data=load_adni_eval(),
        target_column="conversion_3y",
        image_column=IMAGE_COLUMN,
        group_column=GROUP_COLUMN,
        n_splits=n_splits,
        seed=seed,
    )


@register_task
def adni_synthseg_volumes(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    return ColumnTask(
        name="adni_synthseg_volumes",
        kind="regression",
        data=load_adni_eval(),
        target_column="synthseg_volumes",
        image_column=IMAGE_COLUMN,
        group_column=GROUP_COLUMN,
        n_splits=n_splits,
        seed=seed,
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
