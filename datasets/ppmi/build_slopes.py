"""Build clinical target columns for the ppmi-mini eval cohort.

Joins PPMI LONI study-data CSVs onto the 1000 ppmi-mini scans by PATNO and
writes one row per sample_id with baseline + annualized-slope targets.

Output stays OUT of git: PPMI DUA forbids redistributing subject-level data.

    uv run python datasets/ppmi/build_slopes.py \
        --loni-root /mnt/data/medarc/datasets/ppmi \
        --out /mnt/data/medarc/datasets/ppmi/derived/ppmi_mini_clinical.parquet
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

# window for slope fitting, in years relative to the MRI
WINDOW = (-0.25, 4.0)
MIN_VISITS = 2

# Five-domain cognitive battery, the standard PPMI composite (Weintraub et al.
# 2015). Every test is scored so higher = better, so the composite needs no sign
# flips and a negative slope always means decline. Raw totals rather than PPMI's
# derived T-scores: the derived scores are age-normed, which would silently
# remove the very covariate we are trying to hold the model accountable for.
COG_TESTS = {
    "HVLT": ("Hopkins_Verbal_Learning_Test*_*.csv", ["HVLTRT1", "HVLTRT2", "HVLTRT3"]),
    "SDMT": ("Symbol_Digit_Modalities_Test_*.csv", ["SDMTOTAL"]),
    "JLO": ("Benton_Judgement_of_Line_Orientation_*.csv", ["JLO_TOTRAW"]),
    "LNS": ("Letter_-_Number_Sequencing_*.csv", ["LNS_TOTRAW"]),
    "SFT": ("Modified_Semantic_Fluency_*.csv", ["VLTANIM"]),
}
# a visit needs this many of the five to get a composite, else the score jumps
# around as tests drop in and out of the battery
MIN_TESTS = 3


def _find(root: Path, pattern: str) -> Path:
    """LONI stamps every export with its download date, so glob the suffix."""
    hits = sorted(root.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"no match for {pattern} under {root}")
    return hits[-1]


def _visits(path: Path, value_col: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df = df[df[value_col].notna()]
    # INFODT is month/year only -> dates are +/- 2 weeks
    df["dt"] = pd.to_datetime(df["INFODT"], format="%m/%Y", errors="coerce")
    return df.dropna(subset=["dt"])[["PATNO", "dt", value_col]]


def _updrs3(path: Path, *, off_only: bool) -> pd.DataFrame:
    """UPDRS-III motor exam.

    The exam is scored either ON or OFF dopaminergic medication. ON scores are
    drug-suppressed, so an ON/OFF mix measures treatment response as much as
    disease progression. off_only=True keeps OFF exams plus exams on untreated
    participants; off_only=False keeps everything (larger N, mixed states).
    """
    df = pd.read_csv(path, low_memory=False)
    df = df[df["NP3TOT"].notna()]
    if off_only:
        untreated = df["PDSTATE"].isna() & (df["PDTRTMNT"] == 0)
        df = df[(df["PDSTATE"] == "OFF") | untreated]
    # NHY (Hoehn & Yahr) is a 0-5 stage; 101 is an out-of-range "not assessed"
    # code, and 97% of those rows also have no NP3TOT.
    df["NHY"] = df["NHY"].replace(101, np.nan)
    df["dt"] = pd.to_datetime(df["INFODT"], format="%m/%Y", errors="coerce")
    df = df.dropna(subset=["dt"])
    # one exam per person-visit; prefer OFF when a visit has both
    df = df.sort_values(["dt", "PDSTATE"])
    return df.drop_duplicates(["PATNO", "dt"])[["PATNO", "dt", "NP3TOT", "NHY"]]


# DAT-SPECT striatal binding ratios, referenced to occipital white matter.
# Bilateral means only; the file also carries L/R and sub-regional splits.
SBR_REGIONS = {
    "STRIATUM_REF_CWM": "sbr_striatum",
    "PUTAMEN_REF_CWM": "sbr_putamen",
    "CAUDATE_REF_CWM": "sbr_caudate",
}


def _sbr(path: Path) -> pd.DataFrame:
    """One DAT-SPECT per person-scan-date, with the bilateral region SBRs."""
    df = pd.read_csv(path, low_memory=False)
    df["dt"] = pd.to_datetime(df["DATSCAN_DATE"], format="%m/%Y", errors="coerce")
    df = df.dropna(subset=["dt"])
    return df.drop_duplicates(["PATNO", "dt"])[["PATNO", "dt", *SBR_REGIONS]]


def _saa(path: Path) -> pd.DataFrame:
    """CSF alpha-synuclein seed amplification assay status, one row per subject.

    The file has no specimen collection date, only a visit code and the assay
    run date, so a nearest-to-scan match is not possible. SAA status is a
    disease-state marker that rarely reverts, so one value per subject is taken:
    the baseline visit where present, else the earliest visit code available.
    Inconclusive results are dropped rather than coerced either way.
    """
    df = pd.read_csv(path, low_memory=False)
    df = df[df["SAA_Status"].isin(["Positive", "Negative"])]
    df["is_bl"] = (df["CLINICAL_EVENT"] == "BL").astype(int)
    df = df.sort_values(["PATNO", "is_bl", "CLINICAL_EVENT"], ascending=[True, False, True])
    df = df.drop_duplicates("PATNO")
    df["saa_positive"] = (df["SAA_Status"] == "Positive").astype(float)
    return df[["PATNO", "saa_positive"]]


# RBDSQ items 1-12, one binary symptom each.
RBD_ITEMS = [
    "DRMVIVID", "DRMAGRAC", "DRMNOCTB", "SLPLMBMV", "SLPINJUR", "DRMVERBL",
    "DRMFIGHT", "DRMUMV", "DRMOBJFL", "MVAWAKEN", "DRMREMEM", "SLPDSTRB",
]
# item 13 scores 1 if any neurological disorder is present
RBD_NEURO = [
    "STROKE", "HETRA", "PARKISM", "RLS", "NARCLPSY",
    "DEPRS", "EPILEPSY", "BRNINFM", "CNSOTH",
]


def _rbdsq(path: Path) -> pd.DataFrame:
    """REM Sleep Behaviour Disorder Screening Questionnaire total, 0-13.

    PPMI ships the individual items, not the total. Note that PTCGBOTH in this
    file is NOT a score -- it records whether the participant, the caregiver or
    both filled the form in.
    """
    df = pd.read_csv(path, low_memory=False)
    items = df[RBD_ITEMS].sum(axis=1, min_count=len(RBD_ITEMS))
    neuro = df[RBD_NEURO].max(axis=1)  # item 13: any condition present
    df["RBDSQ"] = items + neuro
    df = df[df["RBDSQ"].notna()]
    df["dt"] = pd.to_datetime(df["INFODT"], format="%m/%Y", errors="coerce")
    df = df.dropna(subset=["dt"])
    return df.drop_duplicates(["PATNO", "dt"])[["PATNO", "dt", "RBDSQ"]]


def _cognitive_composite(loni_root: Path) -> pd.DataFrame:
    """Mean z-score across the five-test battery, one row per person-visit.

    Each test is z-scored over every visit in the study before averaging, so the
    tests are on a common scale and the composite is in SD units. MoCA is left
    out on purpose: it is a screening instrument that ceilings in early PD, and
    it already has its own task.
    """
    per_test = []
    for name, (pattern, cols) in COG_TESTS.items():
        path = _find(loni_root / "Non-motor_Assessments", pattern)
        df = pd.read_csv(path, low_memory=False)
        df["raw"] = df[cols].sum(axis=1, min_count=len(cols))
        df = df[df["raw"].notna()]
        df["dt"] = pd.to_datetime(df["INFODT"], format="%m/%Y", errors="coerce")
        df = df.dropna(subset=["dt"])
        # one score per person-visit before z-scoring, else duplicate rows
        # would tilt the mean and SD
        df = df.drop_duplicates(["PATNO", "dt"])
        df["z"] = (df["raw"] - df["raw"].mean()) / df["raw"].std()
        per_test.append(df[["PATNO", "dt", "z"]].assign(test=name))

    long = pd.concat(per_test)
    g = long.groupby(["PATNO", "dt"])["z"]
    comp = g.agg(["mean", "count"]).reset_index()
    comp = comp[comp["count"] >= MIN_TESTS]
    return comp.rename(columns={"mean": "COGCOMP"})[["PATNO", "dt", "COGCOMP"]]


def _slope_and_baseline(
    scans: pd.DataFrame, visits: pd.DataFrame, col: str
) -> pd.DataFrame:
    """Annualized OLS slope over the window, plus the value nearest the scan."""
    m = scans.merge(visits, on="PATNO", how="inner")
    m["yrs"] = (m["dt"] - m["scan"]).dt.days / 365.25
    m = m[m["yrs"].between(*WINDOW)]

    rows = []
    for sid, g in m.groupby("sample_id"):
        nearest = g.loc[g["yrs"].abs().idxmin()]
        slope = np.nan
        if len(g) >= MIN_VISITS and g["yrs"].nunique() >= MIN_VISITS:
            slope = np.polyfit(g["yrs"], g[col], 1)[0]
        rows.append(
            {
                "sample_id": sid,
                f"{col.lower()}_baseline": nearest[col],
                f"{col.lower()}_slope_48m": slope,
                f"{col.lower()}_n_visits": len(g),
            }
        )
    return pd.DataFrame(rows)


def build(loni_root: Path, scans: pd.DataFrame) -> pd.DataFrame:
    subj = loni_root / "_Subject_Characteristics"
    motor = _find(loni_root / "Motor___MDS-UPDRS", "MDS-UPDRS_Part_III_*.csv")
    moca = _find(
        loni_root / "Non-motor_Assessments",
        "Montreal_Cognitive_Assessment__MoCA__*.csv",
    )

    out = scans[["sample_id", "participant_id", "PATNO"]].copy()
    # emit both medication-state variants so the eval can measure whether the
    # ON/OFF confound actually costs anything
    for off_only, tag in ((True, "off"), (False, "all")):
        u = _updrs3(motor, off_only=off_only)
        for col in ("NP3TOT", "NHY"):
            sub = u[u[col].notna()][["PATNO", "dt", col]]
            got = _slope_and_baseline(scans, sub, col)
            got.columns = [
                c if c == "sample_id" else c.replace(col.lower(), f"{col.lower()}_{tag}")
                for c in got.columns
            ]
            out = out.merge(got, on="sample_id", how="left")
    out = out.merge(
        _slope_and_baseline(scans, _visits(moca, "MCATOT"), "MCATOT"),
        on="sample_id",
        how="left",
    )

    # cognitive composite: the primary prognosis target, MoCA without the ceiling
    out = out.merge(
        _slope_and_baseline(scans, _cognitive_composite(loni_root), "COGCOMP"),
        on="sample_id",
        how="left",
    )

    # Function, both instruments. UPDRS-II is patient-reported and Schwab &
    # England is clinician-rated, so agreement between them is evidence the
    # signal is in the participant rather than in one rating style. Neither is
    # scored ON/OFF medication, so no state split is needed.
    for folder, pattern, col in [
        ("Motor___MDS-UPDRS", "MDS_UPDRS_Part_II__Patient_Questionnaire_*.csv", "NP2PTOT"),
        ("Motor___MDS-UPDRS", "Modified_Schwab___England_*.csv", "MSEADLG"),
    ]:
        path = _find(loni_root / folder, pattern)
        out = out.merge(
            _slope_and_baseline(scans, _visits(path, col), col),
            on="sample_id",
            how="left",
        )

    # PD polygenic risk scores. Brain morphology is heritable, so a small true
    # genotype -> structure association is plausible and this is NOT a clean
    # null. The two variants are shipped together on purpose: PPMI enriches its
    # genetic cohort for LRRK2/GBA carriers, recruited at particular sites, so
    # comparing the full score against the LRRK2/GBA-excluded one separates
    # cohort-enrichment effects from the rest of the polygenic signal.
    prs = pd.read_csv(_find(subj, "Polygenic_Risk_Scores_*.csv"), low_memory=False)
    prs = prs.drop_duplicates("PATNO")[
        ["PATNO", "META5_PGS", "META5_excl_LRRK2_GBA_PGS"]
    ].rename(
        columns={
            "META5_PGS": "prs_meta5",
            "META5_excl_LRRK2_GBA_PGS": "prs_meta5_excl_lrrk2_gba",
        }
    )
    out = out.merge(prs, on="PATNO", how="left")

    # UPSIT: 40 scratch-and-sniff items, TOTAL_CORRECT is 0-40, lower = worse
    # smell. Hyposmia is one of the strongest prodromal markers of PD.
    upsit = _find(loni_root / "Non-motor_Assessments", "University_of_Pennsylvania_Smell*_*.csv")
    near = _slope_and_baseline(scans, _visits(upsit, "TOTAL_CORRECT"), "TOTAL_CORRECT")
    out = out.merge(
        near[["sample_id", "total_correct_baseline"]].rename(
            columns={"total_correct_baseline": "upsit_baseline"}
        ),
        on="sample_id",
        how="left",
    )

    rbd = _find(loni_root / "Non-motor_Assessments", "REM_Sleep_Behavior_Disorder_*.csv")
    near = _slope_and_baseline(scans, _rbdsq(rbd), "RBDSQ")
    out = out.merge(
        near[["sample_id", "rbdsq_baseline"]], on="sample_id", how="left"
    )

    # DAT-SPECT SBR. The image itself never reaches the model: these are scalars
    # the imaging core derived from it, so the task is predicting a molecular
    # readout from structural MRI. Same role amyloid/tau play in the ADNI eval.
    sbr = _sbr(_find(loni_root / "Imaging", "Xing_Core_Lab_-_Quant_SBR_*.csv"))
    for col, name in SBR_REGIONS.items():
        sub = sbr[sbr[col].notna()][["PATNO", "dt", col]]
        got = _slope_and_baseline(scans, sub, col)
        got.columns = [
            c if c == "sample_id" else c.replace(col.lower(), name) for c in got.columns
        ]
        out = out.merge(got, on="sample_id", how="left")

    # CSF alpha-synuclein SAA: detects the pathological protein itself rather
    # than its downstream consequences. Near-binary, so expect it to behave like
    # a diagnosis label.
    out = out.merge(
        _saa(_find(loni_root / "Biospecimen", "SAA_Biospecimen_Analysis_Results_*.csv")),
        on="PATNO",
        how="left",
    )

    # Enrollment pathway. Not a target: a confound covariate. PPMI recruits
    # LRRK2/GBA/SNCA carriers into a dedicated genetic cohort, and that
    # membership correlates with PD polygenic risk at r=0.58 by construction. Any
    # target that varies with recruitment needs to be checked against this, which
    # is how the apparent PRS signal turned out to be cohort structure.
    status = pd.read_csv(_find(subj, "Participant_Status_*.csv"), low_memory=False)
    genetic = ["ENRLLRRK2", "ENRLGBA", "ENRLSNCA", "ENRLPINK1", "ENRLPRKN"]
    status = status.drop_duplicates("PATNO")
    status["enrolled_genetic_cohort"] = (
        status[genetic].fillna(0).sum(axis=1) > 0
    ).astype(float)
    for flag, name in (("ENRLHPSM", "enrolled_hyposmia"), ("ENRLRBD", "enrolled_rbd")):
        status[name] = status[flag].fillna(0).astype(float)
    out = out.merge(
        status[["PATNO", "enrolled_genetic_cohort", "enrolled_hyposmia", "enrolled_rbd"]],
        on="PATNO",
        how="left",
    )

    # PATNO is the LONI subject ID and is already encoded in sample_id as
    # sub-<PATNO>_ses-<date>. Keeping it would duplicate the identifier for no
    # gain, so it is dropped before the table is written or published.
    out = out.drop(columns=["PATNO"])

    assert len(out) == len(scans), "join changed row count"
    return out


def load_scans() -> pd.DataFrame:
    from datasets import load_from_disk
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        "medarc/ppmi-mini",
        repo_type="dataset",
        allow_patterns=["dataset_dict.json", "eval/*"],
    )
    df = (
        load_from_disk(path)["eval"]
        .remove_columns(["nifti"])
        .to_pandas()[["sample_id", "participant_id", "scan_date"]]
    )
    df["PATNO"] = df["participant_id"].str.removeprefix("sub-").astype(int)
    df["scan"] = pd.to_datetime(df["scan_date"])
    return df


def _self_check() -> None:
    """Slope of a known line must come back as its gradient."""
    scans = pd.DataFrame(
        {"sample_id": ["s1"], "PATNO": [1], "scan": [pd.Timestamp("2020-01-01")]}
    )
    visits = pd.DataFrame(
        {
            "PATNO": [1, 1, 1],
            "dt": pd.to_datetime(["2020-01-01", "2021-01-01", "2022-01-01"]),
            "X": [10.0, 12.0, 14.0],
        }
    )
    got = _slope_and_baseline(scans, visits, "X")
    assert abs(got["x_slope_48m"].iloc[0] - 2.0) < 0.01, got
    assert got["x_baseline"].iloc[0] == 10.0, got
    # a visit outside the 48m window must be excluded
    visits.loc[3] = [1, pd.Timestamp("2030-01-01"), 99.0]
    assert got["x_n_visits"].iloc[0] == 3
    print("self-check OK")


def cli() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--loni-root", type=Path, default=Path("/mnt/data/medarc/datasets/ppmi"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--self-check", action="store_true")
    args = p.parse_args()

    if args.self_check:
        _self_check()
        return

    scans = load_scans()
    out = build(args.loni_root, scans)
    # This file is meant to be publishable to the gated dataset repo, so the raw
    # LONI subject ID must never reach it -- sample_id already encodes it.
    assert "PATNO" not in out.columns, "PATNO must not reach the output table"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)

    print(f"wrote {args.out}  ({len(out)} rows)")
    for c in sorted(out.columns):
        if c in ("sample_id", "PATNO"):
            continue
        print(f"  {c:28s} n={out[c].notna().sum():4d}/1000")


if __name__ == "__main__":
    cli()
