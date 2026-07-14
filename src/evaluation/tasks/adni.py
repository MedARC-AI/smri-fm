from datasets import Dataset, load_dataset

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

ADNI_EVAL_REPO_ID = "medarc/adni-mini-v1-3"
IMAGE_COLUMN = "nifti"


def load_adni_eval() -> Dataset:
    return load_dataset(ADNI_EVAL_REPO_ID, split="eval")


def _filter_diagnoses(data: Dataset, labels: set[str]) -> Dataset:
    names = data.features["diagnosis"].names
    keep = {names.index(label) for label in labels}
    return data.filter(lambda dx: dx in keep, input_columns="diagnosis")


def _filter_cn_aplus(data: Dataset) -> Dataset:
    data = _filter_diagnoses(data, {"CN"})
    return data.filter(lambda status: status == 1.0, input_columns="amyloid_status")


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
    """Sanity classification (expect near-saturated AUROC)."""
    data = load_adni_eval()
    return ColumnTask(
        name="adni_sex",
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
        metric_fns=(bacc, auroc, auprc),
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
        metric_fns=(bacc, auroc, auprc),
        positive_label=data.features["diagnosis"].names.index("AD"),
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
    """Brain-age gap association (AD cases vs CN-trained age residual)."""
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


# ---------------------------------------------------------------------------
# Primary clinical path / conversion
# ---------------------------------------------------------------------------


@register_task
def adni_mci_conversion_3y(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    """Binary MCI→AD within 36 months (known-at-horizon only)."""
    data = _filter_diagnoses(load_adni_eval(), {"MCI"})
    return ColumnTask(
        name="adni_mci_conversion_3y",
        kind="classification",
        data=data,
        image_column=IMAGE_COLUMN,
        target_column="conversion_3y",
        n_splits=n_splits,
        seed=seed,
        metric_fns=(bacc, auroc, auprc),
        positive_label=1.0,
    )


@register_task
def adni_cn_aplus_to_mci_ad_3y(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    """Secondary: CN A+ → MCI/AD within 36 months (thin events)."""
    data = _filter_cn_aplus(load_adni_eval())
    return ColumnTask(
        name="adni_cn_aplus_to_mci_ad_3y",
        kind="classification",
        data=data,
        image_column=IMAGE_COLUMN,
        target_column="cn_aplus_to_mci_ad_3y",
        n_splits=n_splits,
        seed=seed,
        metric_fns=(bacc, auroc, auprc),
        positive_label=1.0,
    )


# ---------------------------------------------------------------------------
# Primary cognition (continuous annualized slopes, 48m)
# ---------------------------------------------------------------------------


@register_task
def adni_adas13_slope_48m(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    """Annualized ADAS-Cog13 slope (higher = worse multi-domain cognition)."""
    return ColumnTask(
        name="adni_adas13_slope_48m",
        kind="regression",
        data=load_adni_eval(),
        image_column=IMAGE_COLUMN,
        target_column="adas13_slope_48m",
        n_splits=n_splits,
        seed=seed,
        metric_fns=(pearson_r, spearman_r, r2),
    )


@register_task
def adni_ravlt_delayed_slope_48m(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    """Annualized RAVLT 30-minute delayed-recall slope (more negative = decline)."""
    return ColumnTask(
        name="adni_ravlt_delayed_slope_48m",
        kind="regression",
        data=load_adni_eval(),
        image_column=IMAGE_COLUMN,
        target_column="ravlt_delayed_slope_48m",
        n_splits=n_splits,
        seed=seed,
        metric_fns=(pearson_r, spearman_r, r2),
    )


@register_task
def adni_ravlt_learning_slope_48m(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    """Annualized RAVLT learning slope (sum of AVTOT1-5; more negative = decline)."""
    return ColumnTask(
        name="adni_ravlt_learning_slope_48m",
        kind="regression",
        data=load_adni_eval(),
        image_column=IMAGE_COLUMN,
        target_column="ravlt_learning_slope_48m",
        n_splits=n_splits,
        seed=seed,
        metric_fns=(pearson_r, spearman_r, r2),
    )


# ---------------------------------------------------------------------------
# Primary atrophy (SynthSeg ICV-normalized annualized slopes, 48m)
# ---------------------------------------------------------------------------


@register_task
def adni_hippocampus_slope_48m(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    """Hippocampus L+R / ICV annualized change (more negative = atrophy)."""
    return ColumnTask(
        name="adni_hippocampus_slope_48m",
        kind="regression",
        data=load_adni_eval(),
        image_column=IMAGE_COLUMN,
        target_column="hippocampus_slope_48m",
        n_splits=n_splits,
        seed=seed,
        metric_fns=(pearson_r, spearman_r, r2),
    )


@register_task
def adni_ventricle_slope_48m(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    """Lateral ventricles L+R / ICV annualized change (more positive = expansion)."""
    return ColumnTask(
        name="adni_ventricle_slope_48m",
        kind="regression",
        data=load_adni_eval(),
        image_column=IMAGE_COLUMN,
        target_column="ventricle_slope_48m",
        n_splits=n_splits,
        seed=seed,
        metric_fns=(pearson_r, spearman_r, r2),
    )


@register_task
def adni_entorhinal_thickness_slope_48m(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    """Bilateral entorhinal cortical-thickness annualized change."""
    return ColumnTask(
        name="adni_entorhinal_thickness_slope_48m",
        kind="regression",
        data=load_adni_eval(),
        image_column=IMAGE_COLUMN,
        target_column="entorhinal_thickness_slope_48m",
        n_splits=n_splits,
        seed=seed,
        metric_fns=(pearson_r, spearman_r, r2),
    )


# ---------------------------------------------------------------------------
# CN amyloid-positive prognosis (continuous annualized slopes, 48m)
# ---------------------------------------------------------------------------


def _cn_aplus_slope_task(
    *,
    name: str,
    target_column: str,
    n_splits: int,
    seed: int,
) -> ColumnTask:
    data = _filter_cn_aplus(load_adni_eval())
    return ColumnTask(
        name=name,
        kind="regression",
        data=data,
        image_column=IMAGE_COLUMN,
        target_column=target_column,
        n_splits=n_splits,
        seed=seed,
        metric_fns=(pearson_r, spearman_r, r2),
    )


@register_task
def adni_cn_aplus_adas13_slope_48m(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    return _cn_aplus_slope_task(
        name="adni_cn_aplus_adas13_slope_48m",
        target_column="adas13_slope_48m",
        n_splits=n_splits,
        seed=seed,
    )


@register_task
def adni_cn_aplus_ravlt_delayed_slope_48m(
    n_splits: int = 5, seed: int = 0
) -> ColumnTask:
    return _cn_aplus_slope_task(
        name="adni_cn_aplus_ravlt_delayed_slope_48m",
        target_column="ravlt_delayed_slope_48m",
        n_splits=n_splits,
        seed=seed,
    )


@register_task
def adni_cn_aplus_ravlt_learning_slope_48m(
    n_splits: int = 5, seed: int = 0
) -> ColumnTask:
    return _cn_aplus_slope_task(
        name="adni_cn_aplus_ravlt_learning_slope_48m",
        target_column="ravlt_learning_slope_48m",
        n_splits=n_splits,
        seed=seed,
    )


@register_task
def adni_cn_aplus_hippocampus_slope_48m(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    return _cn_aplus_slope_task(
        name="adni_cn_aplus_hippocampus_slope_48m",
        target_column="hippocampus_slope_48m",
        n_splits=n_splits,
        seed=seed,
    )


@register_task
def adni_cn_aplus_ventricle_slope_48m(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    return _cn_aplus_slope_task(
        name="adni_cn_aplus_ventricle_slope_48m",
        target_column="ventricle_slope_48m",
        n_splits=n_splits,
        seed=seed,
    )


@register_task
def adni_cn_aplus_entorhinal_thickness_slope_48m(
    n_splits: int = 5, seed: int = 0
) -> ColumnTask:
    return _cn_aplus_slope_task(
        name="adni_cn_aplus_entorhinal_thickness_slope_48m",
        target_column="entorhinal_thickness_slope_48m",
        n_splits=n_splits,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Secondary molecular association 
# ---------------------------------------------------------------------------


@register_task
def adni_amyloid_centiloid(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    """Secondary: PET amyloid burden (centiloids), all diagnoses."""
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
def adni_tau_suvr(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    """Secondary: meta-temporal tau PET SUVR, all diagnoses."""
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
def adni_csf_abeta(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    """Secondary: CSF Aβ42 (Elecsys)."""
    return ColumnTask(
        name="adni_csf_abeta",
        kind="regression",
        data=load_adni_eval(),
        image_column=IMAGE_COLUMN,
        target_column="csf_abeta",
        n_splits=n_splits,
        seed=seed,
        metric_fns=(spearman_r, pearson_r),
    )


@register_task
def adni_csf_ptau(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    """Secondary: CSF p-tau."""
    return ColumnTask(
        name="adni_csf_ptau",
        kind="regression",
        data=load_adni_eval(),
        image_column=IMAGE_COLUMN,
        target_column="csf_ptau",
        n_splits=n_splits,
        seed=seed,
        metric_fns=(spearman_r, pearson_r),
    )


@register_task
def adni_csf_ttau(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    """Secondary: CSF total tau."""
    return ColumnTask(
        name="adni_csf_ttau",
        kind="regression",
        data=load_adni_eval(),
        image_column=IMAGE_COLUMN,
        target_column="csf_ttau",
        n_splits=n_splits,
        seed=seed,
        metric_fns=(spearman_r, pearson_r),
    )
