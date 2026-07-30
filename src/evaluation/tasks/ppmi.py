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

# Clinical targets live in the gated dataset repo alongside the imaging. Set
# PPMI_CLINICAL_PARQUET to override with a locally built file, which is what you
# want when iterating on datasets/ppmi/build_slopes.py.
CLINICAL_IN_REPO = "clinical/ppmi_mini_clinical.parquet"
CLINICAL_PARQUET = os.environ.get("PPMI_CLINICAL_PARQUET")


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
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    if CLINICAL_PARQUET:
        source = CLINICAL_PARQUET
    else:
        try:
            source = hf_hub_download(
                PPMI_EVAL_REPO_ID, CLINICAL_IN_REPO, repo_type="dataset"
            )
        except EntryNotFoundError as exc:
            raise FileNotFoundError(
                f"{CLINICAL_IN_REPO} is not in {PPMI_EVAL_REPO_ID} yet. Build it "
                "locally with datasets/ppmi/build_slopes.py and point "
                "PPMI_CLINICAL_PARQUET at the result."
            ) from exc
    data = load_ppmi_eval()
    clinical = pd.read_parquet(source).set_index("sample_id")
    missing = set(data["sample_id"]) - set(clinical.index)
    if missing:
        raise ValueError(f"{len(missing)} samples absent from {source}; rebuild it")
    clinical = clinical.loc[list(data["sample_id"])]
    for col in clinical.columns:
        # the sidecar repeats participant_id as a grouping key; the split already
        # has it, and add_column rejects duplicates
        if col in data.column_names:
            continue
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
def ppmi_diagnosis_3way(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    """3-way CN / Prodromal / PD, the clinically meaningful ordering.

    SWEDD is dropped. Those participants carry a PD diagnosis but a normal DAT
    scan, so the label is contested by construction and a known noise source in
    PD imaging studies. Chance is 33% balanced accuracy.
    """
    data = _filter_diagnoses(load_ppmi_eval(), {"CN", "Prodromal", "PD"})
    return ColumnTask(
        name="ppmi_diagnosis_3way",
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
def ppmi_updrs3_slope_48m_off(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    """Annualized MDS-UPDRS Part III motor progression. Higher = worsening.

    OFF-medication and untreated exams only (n=765). Cleaner signal, smaller N.
    Compare against ppmi_updrs3_slope_48m_all to price the ON/OFF confound.
    """
    return ColumnTask(
        name="ppmi_updrs3_slope_48m_off",
        kind="regression",
        data=load_ppmi_clinical(),
        image_column=IMAGE_COLUMN,
        target_column="np3tot_off_slope_48m",
        n_splits=n_splits,
        seed=seed,
        metric_fns=(pearson_r, spearman_r, r2),
    )


@register_task
def ppmi_updrs3_slope_48m_all(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    """As above but keeping ON-medication exams too (n=953).

    Larger N, but the target mixes disease progression with treatment response.
    """
    return ColumnTask(
        name="ppmi_updrs3_slope_48m_all",
        kind="regression",
        data=load_ppmi_clinical(),
        image_column=IMAGE_COLUMN,
        target_column="np3tot_all_slope_48m",
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
        target_column="nhy_off_slope_48m",
        n_splits=n_splits,
        seed=seed,
        metric_fns=(pearson_r, spearman_r, r2),
    )


# ---------------------------------------------------------------------------
# Severity within PD only.
#
# The whole-cohort baseline tasks below are near-collinear with diagnosis (CN
# and Prodromal sit at H&Y 0, PD at 1-2), so a good score there can just mean
# "detects PD", not "grades severity". Holding diagnosis constant separates the
# two: signal that survives here is severity, signal that vanishes was group
# separation.
# ---------------------------------------------------------------------------


@register_task
def ppmi_updrs3_baseline_pd(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    """UPDRS-III motor score within PD only (n=346, mean 23.1 +- 10.4)."""
    return ColumnTask(
        name="ppmi_updrs3_baseline_pd",
        kind="regression",
        data=_filter_diagnoses(load_ppmi_clinical(), {"PD"}),
        image_column=IMAGE_COLUMN,
        target_column="np3tot_off_baseline",
        n_splits=n_splits,
        seed=seed,
        metric_fns=(pearson_r, spearman_r, r2),
    )


@register_task
def ppmi_hoehn_yahr_baseline_pd(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    """H&Y stage within PD only (n=346). Nearly all stage 1 or 2, so the
    achievable ceiling is low even if severity is readable."""
    return ColumnTask(
        name="ppmi_hoehn_yahr_baseline_pd",
        kind="regression",
        data=_filter_diagnoses(load_ppmi_clinical(), {"PD"}),
        image_column=IMAGE_COLUMN,
        target_column="nhy_off_baseline",
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
        target_column="np3tot_off_baseline",
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
        target_column="nhy_off_baseline",
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


# ---------------------------------------------------------------------------
# Cognitive composite, function, and a genetic negative control
# ---------------------------------------------------------------------------


@register_task
def ppmi_cogcomp_slope_48m(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    """Annualized slope of the 5-test cognitive composite. Lower = declining.

    Mean z-score across HVLT-R, Symbol Digit, Benton JLO, Letter-Number
    Sequencing and semantic fluency. The primary cognitive prognosis target:
    MoCA is a screening instrument that ceilings in early PD, so a composite
    built from full-length tests has a wider dynamic range at the same N
    (n=930 vs 921).
    """
    return ColumnTask(
        name="ppmi_cogcomp_slope_48m",
        kind="regression",
        data=load_ppmi_clinical(),
        image_column=IMAGE_COLUMN,
        target_column="cogcomp_slope_48m",
        n_splits=n_splits,
        seed=seed,
        metric_fns=(pearson_r, spearman_r, r2),
    )


@register_task
def ppmi_cogcomp_baseline(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    """Cross-sectional cognitive composite, in SD units."""
    return ColumnTask(
        name="ppmi_cogcomp_baseline",
        kind="regression",
        data=load_ppmi_clinical(),
        image_column=IMAGE_COLUMN,
        target_column="cogcomp_baseline",
        n_splits=n_splits,
        seed=seed,
        metric_fns=(pearson_r, spearman_r, r2),
    )


@register_task
def ppmi_updrs2_slope_48m(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    """Annualized MDS-UPDRS Part II slope. Higher = worsening.

    Patient-reported motor experiences of daily living. Unlike Part III this is
    not scored ON/OFF medication, so it needs no state split.
    """
    return ColumnTask(
        name="ppmi_updrs2_slope_48m",
        kind="regression",
        data=load_ppmi_clinical(),
        image_column=IMAGE_COLUMN,
        target_column="np2ptot_slope_48m",
        n_splits=n_splits,
        seed=seed,
        metric_fns=(pearson_r, spearman_r, r2),
    )


@register_task
def ppmi_schwab_england_slope_48m(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    """Annualized Schwab & England ADL slope. Lower = declining independence.

    Clinician-rated, so it measures the same construct as UPDRS-II through a
    different rater. Agreement between the two is evidence that any signal is
    in the participant rather than in one rating style.
    """
    return ColumnTask(
        name="ppmi_schwab_england_slope_48m",
        kind="regression",
        data=load_ppmi_clinical(),
        image_column=IMAGE_COLUMN,
        target_column="mseadlg_slope_48m",
        n_splits=n_splits,
        seed=seed,
        metric_fns=(pearson_r, spearman_r, r2),
    )


@register_task
def ppmi_prs(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    """PD polygenic risk score (META5).

    Brain morphology is heritable, so genotype -> structure is a real causal
    route and a small association here would not be surprising. What is
    surprising is the size: ViT-B reaches r ~ 0.17 and ViT-L ~ 0.21 on n=827,
    far above what the PRS/imaging literature reports at this sample size.

    Run alongside ppmi_prs_excl_lrrk2_gba to separate the two explanations.
    """
    return ColumnTask(
        name="ppmi_prs",
        kind="regression",
        data=load_ppmi_clinical(),
        image_column=IMAGE_COLUMN,
        target_column="prs_meta5",
        n_splits=n_splits,
        seed=seed,
        metric_fns=(pearson_r, spearman_r, r2),
    )


@register_task
def ppmi_prs_excl_lrrk2_gba(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    """Same score with the LRRK2 and GBA loci removed.

    PPMI enriches its genetic cohort for LRRK2/GBA carriers, who are recruited
    at particular sites and are disproportionately of Ashkenazi ancestry. If the
    signal is cohort structure rather than polygenic biology, dropping those
    loci should cost most of it. Measured: 0.168 -> 0.099 for ViT-B, a 41% drop,
    while the two scores still correlate at 0.65 with each other. So much of the
    association travels with cohort enrichment, but not all of it.
    """
    return ColumnTask(
        name="ppmi_prs_excl_lrrk2_gba",
        kind="regression",
        data=load_ppmi_clinical(),
        image_column=IMAGE_COLUMN,
        target_column="prs_meta5_excl_lrrk2_gba",
        n_splits=n_splits,
        seed=seed,
        metric_fns=(pearson_r, spearman_r, r2),
    )


# ---------------------------------------------------------------------------
# Molecular and dopaminergic targets: predicted from T1w, measured elsewhere
# ---------------------------------------------------------------------------


def _sbr_task(name: str, column: str, doc: str, n_splits: int, seed: int) -> ColumnTask:
    task = ColumnTask(
        name=name,
        kind="regression",
        data=load_ppmi_clinical(),
        image_column=IMAGE_COLUMN,
        target_column=column,
        n_splits=n_splits,
        seed=seed,
        metric_fns=(pearson_r, spearman_r, r2),
    )
    task.__doc__ = doc
    return task


@register_task
def ppmi_sbr_striatum(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    """DAT-SPECT striatal binding ratio, whole striatum. Lower = more loss.

    The SPECT image never reaches the model. SBR is a scalar the imaging core
    derives from it by referencing striatal counts against occipital white
    matter, so this asks whether a T1w MRI carries information about presynaptic
    dopaminergic terminal density. Direct analog of the ADNI amyloid/tau tasks.

    Expected to be near zero: nigrostriatal degeneration is a loss of
    presynaptic terminals, which is why DAT-SPECT rather than MRI is the PD
    imaging biomarker in the first place.
    """
    return _sbr_task(
        "ppmi_sbr_striatum", "sbr_striatum_baseline", ppmi_sbr_striatum.__doc__, n_splits, seed
    )


@register_task
def ppmi_sbr_putamen(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    """SBR in the putamen, where dopaminergic loss appears earliest in PD."""
    return _sbr_task(
        "ppmi_sbr_putamen", "sbr_putamen_baseline", ppmi_sbr_putamen.__doc__, n_splits, seed
    )


@register_task
def ppmi_saa(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    """CSF alpha-synuclein seed amplification assay status, positive vs negative.

    SAA detects misfolded alpha-synuclein itself rather than its downstream
    consequences, which makes it the strongest molecular diagnostic in PD. It is
    also near-binary and near-collinear with diagnosis, so read this against
    ppmi_pd_cn rather than in isolation: a score close to that one means the
    model is recovering diagnosis, not the molecular state.
    """
    return ColumnTask(
        name="ppmi_saa",
        kind="classification",
        data=load_ppmi_clinical(),
        image_column=IMAGE_COLUMN,
        target_column="saa_positive",
        n_splits=n_splits,
        seed=seed,
        metric_fns=(bacc, auroc, auprc),
        positive_label=1.0,
    )
