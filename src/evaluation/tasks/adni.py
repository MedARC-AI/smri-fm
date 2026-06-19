import datasets as hfds
from datasets import Dataset, load_dataset
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from evaluation.tasks.brain_age_gap import BrainAgeGapTask
from evaluation.tasks.column import ColumnTask
from evaluation.tasks.registry import register_task

REPO_ID = "medarc/adni_eval"
GROUP_COLUMN = "participant_id"
IMAGE_COLUMN = "nifti"


def load_adni_eval_pooled() -> Dataset:
    """Pool all HF splits into one dataset, keeping every scan session.

    Subject leakage is prevented downstream by grouped CV on ``participant_id``
    rather than by dropping repeated sessions.
    """
    dataset_dict = load_dataset(REPO_ID)
    return hfds.concatenate_datasets(list(dataset_dict.values()))


def load_adni_ad_cn_pooled() -> Dataset:
    """Pooled ADNI restricted to the binary AD-vs-CN contrast.

    ``diagnosis`` is a 3-class label (CN/MCI/AD); the binary task drops MCI so it is
    a genuine AD-vs-CN classification. Labels are kept as their native class ids
    (CN, AD); the accuracy / balanced-accuracy metrics are label-agnostic, so no
    remap to {0, 1} is needed. Filtering is index-based and does not rewrite images.
    """
    data = load_adni_eval_pooled()
    names = data.features["diagnosis"].names
    keep = {names.index("CN"), names.index("AD")}
    return data.filter(lambda dx: dx in keep, input_columns="diagnosis")


@register_task
def adni_age(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    return ColumnTask(
        name="adni_age",
        kind="regression",
        data=load_adni_eval_pooled(),
        splitter=GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed),
        target_column="age",
        image_column=IMAGE_COLUMN,
        group_column=GROUP_COLUMN,
    )


@register_task
def adni_sex(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    return ColumnTask(
        name="adni_sex",
        kind="classification",
        data=load_adni_eval_pooled(),
        splitter=StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed),
        target_column="sex",
        image_column=IMAGE_COLUMN,
        group_column=GROUP_COLUMN,
    )


@register_task
def adni_ad_cn(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    """Binary AD-vs-CN diagnosis classification (MCI dropped)."""
    return ColumnTask(
        name="adni_ad_cn",
        kind="classification",
        data=load_adni_ad_cn_pooled(),
        splitter=StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed),
        target_column="diagnosis",
        image_column=IMAGE_COLUMN,
        group_column=GROUP_COLUMN,
    )


@register_task
def adni_cn_mci_ad(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    """3-way diagnosis classification over all CN / MCI / AD scans."""
    return ColumnTask(
        name="adni_cn_mci_ad",
        kind="classification",
        data=load_adni_eval_pooled(),
        splitter=StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed),
        target_column="diagnosis",
        image_column=IMAGE_COLUMN,
        group_column=GROUP_COLUMN,
    )


@register_task
def adni_synthseg_volumes(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    return ColumnTask(
        name="adni_synthseg_volumes",
        kind="regression",
        data=load_adni_eval_pooled(),
        splitter=GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed),
        target_column="synthseg_volumes",
        image_column=IMAGE_COLUMN,
        group_column=GROUP_COLUMN,
    )


@register_task
def adni_ad_cn_bag() -> BrainAgeGapTask:
    return BrainAgeGapTask(
        name="adni_ad_cn_bag",
        data=load_adni_eval_pooled(),
        age_column="age",
        dx_column="diagnosis",
        control_label=0,
        case_label=2,
        image_column=IMAGE_COLUMN,
        group_column=GROUP_COLUMN,
    )
