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


def _updrs3_offstate(path: Path) -> pd.DataFrame:
    """UPDRS-III is scored ON or OFF medication; ON scores are drug-suppressed.

    Keep OFF exams and exams on untreated participants, drop ON.
    """
    df = pd.read_csv(path, low_memory=False)
    df = df[df["NP3TOT"].notna()]
    untreated = df["PDSTATE"].isna() & (df["PDTRTMNT"] == 0)
    df = df[(df["PDSTATE"] == "OFF") | untreated]
    df["NHY"] = df["NHY"].replace(101, np.nan)  # 101 = "unable to assess"
    df["dt"] = pd.to_datetime(df["INFODT"], format="%m/%Y", errors="coerce")
    df = df.dropna(subset=["dt"])
    # one exam per person-visit
    return df.sort_values("dt").drop_duplicates(["PATNO", "dt"])[
        ["PATNO", "dt", "NP3TOT", "NHY"]
    ]


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

    out = scans[["sample_id", "PATNO"]].copy()
    u = _updrs3_offstate(motor)
    for col in ("NP3TOT", "NHY"):
        sub = u[u[col].notna()][["PATNO", "dt", col]]
        out = out.merge(_slope_and_baseline(scans, sub, col), on="sample_id", how="left")
    out = out.merge(
        _slope_and_baseline(scans, _visits(moca, "MCATOT"), "MCATOT"),
        on="sample_id",
        how="left",
    )

    # UPSIT (smell) and RBD (dream enactment) are single-timepoint prodromal markers
    for folder, pattern, col in [
        ("Non-motor_Assessments", "University_of_Pennsylvania_Smell*_*.csv", "TOTAL_CORRECT"),
        ("Non-motor_Assessments", "REM_Sleep_Behavior_Disorder_*.csv", "PTCGBOTH"),
    ]:
        try:
            path = _find(loni_root / folder, pattern)
        except FileNotFoundError:
            continue
        df = pd.read_csv(path, low_memory=False)
        if col not in df.columns:
            continue
        near = _slope_and_baseline(scans, _visits(path, col), col)
        out = out.merge(near[["sample_id", f"{col.lower()}_baseline"]], on="sample_id", how="left")

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
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)

    print(f"wrote {args.out}  ({len(out)} rows)")
    for c in sorted(out.columns):
        if c in ("sample_id", "PATNO"):
            continue
        print(f"  {c:28s} n={out[c].notna().sum():4d}/1000")


if __name__ == "__main__":
    cli()
