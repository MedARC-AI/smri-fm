# ADNI evaluation dataset

`curate.py` reproduces the ADNI v1 evaluation cohort published at
`medarc/adni_eval`. It matches processed scans to the raw ADNI study-data tables
(PTDEMOG, DXSUM, UPENN Roche Elecsys CSF) and the UC Berkeley amyloid/tau PET
tables, applies quality and demographic filters, attaches the nine v1 task labels,
assigns one subject-exclusive split, and writes Arrow shards with embedded NIfTI
bytes. ADNIMERGE (the stale derived table) is not used.

This script is not used during evaluation. Evaluation loads the published dataset
directly with `load_dataset("medarc/adni_eval", token=True)` and pools the splits,
running grouped CV on `participant_id`.

```bash
uv run python datasets/adni/curate.py \
  --data-root /path/to/adni \
  --demog-csv /path/to/PTDEMOG.csv \
  --dxsum-csv /path/to/DXSUM.csv \
  --csf-csv /path/to/UPENNBIOMK_ROCHE_ELECSYS.csv \
  --amyloid-csv /path/to/UCBERKELEY_AMY_6MM.csv \
  --tau-csv /path/to/UCBERKELEY_TAU_6MM.csv \
  --output-dir /path/to/adni-smri \
  --num-proc 8 --max-shard-size 1GB
```

## V1 eval suite

Nine biomarker-faithful tasks are predicted from a single T1w scan. Every scan
carries the labels for all tasks it qualifies for (NaN otherwise); each task
auto-filters to the scans with its label present (`ColumnTask` drops NaN targets),
so the per-task cohorts differ. A single subject-exclusive split is shipped;
grouped cross-validation on `participant_id` is done at eval time.

Tasks are defined in `src/evaluation/tasks/adni.py`. What is deliberately not in
v1 and why is documented in `datasets/adni/v1_excluded.md`.

## Sources

Labels come from the raw ADNI study-data tables.

| Source | File | Provides |
|---|---|---|
| Demographics | `PTDEMOG_*.csv` | sex (`PTGENDER`), DOB (`PTDOB`) -> exact age at scan |
| Diagnosis | `DXSUM_*.csv` | `DIAGNOSIS` 1=CN / 2=MCI / 3=Dementia, per visit |
| CSF (Roche Elecsys) | `UPENNBIOMK_ROCHE_ELECSYS_*.csv` | `ABETA42`, `TAU`, `PTAU` |
| UC Berkeley amyloid | `UCBERKELEY_AMY_6MM_*.csv` | `AMYLOID_STATUS`, `CENTILOIDS`, `SUMMARY_SUVR` |
| UC Berkeley tau | `UCBERKELEY_TAU_6MM_*.csv` | `META_TEMPORAL_SUVR` |

Scans are joined to the clinical tables by `PTID` (underscore-stripped) and to the
PET tables by `RID`. Diagnosis, CSF, and PET are each paired to the nearest
visit/scan within +/-365 days; the gap is recorded (`clinical_match_days`,
`csf_match_days`, `*_match_days`) for filtering. Age is computed per scan as
`(scan_date - PTDOB) / 365.25`.

## Tasks

| # | Task name | Column | Kind | Label definition |
|---|---|---|---|---|
| 1 | `adni_ad_cn` | `diagnosis` | classification | DXSUM `DIAGNOSIS`, AD vs CN (MCI dropped) |
| 2 | `adni_amyloid_status` | `amyloid_status` | classification | UC Berkeley precomputed positivity (FBP SUVR>1.11, FBB>1.08) |
| 3 | `adni_amyloid_centiloid` | `amyloid_centiloid` | regression | Centiloid amyloid burden |
| 4 | `adni_tau_status` | `tau_status` | classification | meta-temporal SUVR > 1.23 (Jack 2017) |
| 5 | `adni_tau_suvr` | `tau_suvr` | regression | meta-temporal SUVR |
| 6 | `adni_csf_abeta` | `csf_abeta` | regression | CSF Aβ42 pg/mL (Elecsys) |
| 7 | `adni_csf_ptau` | `csf_ptau` | regression | CSF p-tau pg/mL (Elecsys) |
| 8 | `adni_csf_ttau` | `csf_ttau` | regression | CSF t-tau pg/mL (Elecsys) |
| 9 | `adni_mci_conversion` | `conversion_3y` | classification | MCI-at-scan converts to AD within 36 mo |

## Metrics

- Sex canary: balanced accuracy + AUROC; fixed regularization, never used for
  model selection.
- AD vs CN: AUROC primary; AUPRC and balanced accuracy secondary.
- Brain age: Pearson r and R2 primary; MAE in years after train-fold-fitted
  age-bias correction secondary.
- SynthSeg volumes: per-region R2 and Pearson correlation.
- BAG: uses the same bias-corrected age-head metrics, plus the case-control BAG
  t-statistic.

## Target scale

Continuous biomarker regressions (`adni_amyloid_centiloid`, `adni_tau_suvr`,
`adni_csf_abeta`, `adni_csf_ptau`, `adni_csf_ttau`) are evaluated on the raw target
scale unless otherwise noted. These targets can be skewed, thresholded,
assay-clipped, or biologically nonlinear, so raw-scale R2 and MAE may be
tail-sensitive. Spearman correlation is rank-based and less sensitive to this.

V1 task names should not silently transform targets. If transformed targets are
useful, add explicit variants such as `adni_tau_suvr_log`,
`adni_amyloid_centiloid_log1p`, `adni_csf_ptau_log`, or `adni_csf_ttau_log`.
Distribution-fit transforms such as rank or inverse-normal transforms must be fit
inside each training fold only to avoid target-distribution leakage.

## Label notes

- Diagnosis is a 3-class ClassLabel `CN/MCI/AD` (DXSUM `DIAGNOSIS` 3=Dementia ->
  `AD`). `adni_ad_cn` drops MCI; `adni_cn_mci_ad` keeps all three. DXSUM covers
  3,071/3,110 MRI subjects; the ~39 with no diagnosis are dropped (ClassLabel
  needs a value, and they carry no AD-axis label for any task).
- Amyloid positivity uses UC Berkeley's precomputed `AMYLOID_STATUS`, whose
  effective cutoff is tracer-specific summary SUVR (FBP > 1.11, FBB > 1.08,
  approximately Centiloid 20). Amyloid burden uses Centiloids.
- CSF censoring: Elecsys can rail values as `>1700` / `<200`; the parser clips to
  the boundary. The raw UPENN Elecsys export is already fully numeric, so no
  clipping is currently triggered.
- Tau positivity is derived (no precomputed status ships with the tau table); the
  1.23 meta-temporal SUVR cutoff (Jack 2017) is a documented choice. The tau
  burden regression is threshold-free.
- MCI -> AD conversion is prognostic but single-scan-input: the at-risk set is
  scans whose diagnosis is MCI; `conversion_3y` = 1 if AD is reached within 36 mo,
  0 if followed >=36 mo without converting, NaN (excluded) if censored earlier.
  Raw `conversion_event` / `conversion_time_months` are also stored for a future
  time-to-event (C-index) task.

## Approximate cohorts

With seed 4466 and +/-365 day matching windows, the cohort has about 7,804 scans
and 2,406 subjects total. Per task: AD-vs-CN 3,030 CN / 1,871 AD; amyloid 5,067
scans (2,570 positive); tau 1,653 scans (735 positive); CSF about 4,127 scans;
MCI -> AD 1,811 scans (608 positive). Exact counts are in `manifest_report.json`.