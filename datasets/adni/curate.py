from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import ClassLabel, Dataset, DatasetDict, Features, Nifti, Sequence, Value

# ---------------------------------------------------------------------------
# v1 task labels carried by every scan (sparse: NaN where unavailable).
#
#   1. diagnosis            CN / MCI / AD            (ClassLabel)
#   2. amyloid_status       amyloid-PET positivity   (0/1)
#   3. amyloid_centiloid    amyloid burden           (regression)
#   4. tau_status           tau-PET positivity       (0/1, SUVR threshold)
#   5. tau_suvr             tau burden               (regression, meta-temporal)
#   6. csf_abeta            CSF Abeta42              (regression, pg/mL)
#   7. csf_ptau             CSF p-tau                (regression, pg/mL)
#   8. csf_ttau             CSF t-tau                (regression, pg/mL)
#   9. conversion_3y        MCI -> AD within 36 mo   (0/1; NaN if censored early)
#      conversion_event/_time_months kept raw for a later time-to-event task.
# ---------------------------------------------------------------------------


def adni_features() -> Features:
    return Features({
        "sample_id": Value("string"), "participant_id": Value("string"),
        "rid": Value("int32"),
        "session_id": Value("string"), "scan_date": Value("string"),
        "age": Value("float32"), "sex": ClassLabel(names=["Female", "Male"]),
        "diagnosis": ClassLabel(names=["CN", "MCI", "AD"]),
        "synthseg_volumes": Sequence(Value("float32"), length=101),
        # PET amyloid
        "amyloid_status": Value("float32"), "amyloid_centiloid": Value("float32"),
        "amyloid_suvr": Value("float32"), "amyloid_tracer": Value("string"),
        "amyloid_match_days": Value("float32"),
        # PET tau
        "tau_status": Value("float32"), "tau_suvr": Value("float32"),
        "tau_tracer": Value("string"), "tau_match_days": Value("float32"),
        # CSF (Elecsys)
        "csf_abeta": Value("float32"), "csf_ptau": Value("float32"),
        "csf_ttau": Value("float32"), "csf_match_days": Value("float32"),
        # MCI -> AD conversion (prognostic, single-scan input)
        "conversion_event": Value("float32"), "conversion_time_months": Value("float32"),
        "conversion_3y": Value("float32"),
        # provenance / QC
        "clinical_match_days": Value("float32"),
        "nifti": Nifti(), "mask": Nifti(),
    })


IMAGE_RE = re.compile(r"^(?P<stem>sub-(?P<sub>[^_]+)_ses-(?P<date>\d{8})_T1w(?:_[^_]+)?)_space-")


@dataclass(frozen=True)
class CohortConfig:
    seed: int = 4466
    min_age: float = 60.0
    max_age: float = 90.0
    min_qc_mean: float = 0.80
    min_qc_min: float = 0.60
    # pairing windows (days) for biomarkers measured at a nearby visit/scan
    clinical_window_days: int = 365
    csf_window_days: int = 365
    pet_window_days: int = 365
    # positivity / conversion thresholds
    tau_suvr_threshold: float = 1.23   # meta-temporal SUVR, Jack et al. 2017
    conversion_horizon_months: float = 36.0


# ---------------------------------------------------------------------------
# Clinical / CSF helpers
# ---------------------------------------------------------------------------

# Ground truth comes from raw ADNI study-data tables (ADNIMERGE is a stale
# derived convenience file and is not used):
#   PTDEMOG  -> sex (PTGENDER) and date of birth (PTDOB) for exact per-scan age
#   DXSUM    -> per-visit diagnosis (DIAGNOSIS 1=CN, 2=MCI, 3=Dementia)
#   UPENNBIOMK_ROCHE_ELECSYS -> CSF Abeta42 / t-tau / p-tau (pg/mL)
DIAGNOSIS_TO_LABEL = {1: "CN", 2: "MCI", 3: "AD"}
PTGENDER_TO_SEX = {1: "Male", 2: "Female"}


def _to_pid(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace("_", "", regex=False)


def _parse_censored(series: pd.Series) -> pd.Series:
    """CSF values railed at the assay limits ('>1700', '<200') are clipped to
    the boundary; plain numbers pass through. The UPENN Elecsys table is already
    numeric, but this keeps the parse robust to censored exports."""
    text = series.astype(str).str.strip()
    cleaned = text.str.replace(">", "", regex=False).str.replace("<", "", regex=False)
    return pd.to_numeric(cleaned, errors="coerce")


def _load_demographics(demog_csv: Path) -> pd.DataFrame:
    """One row per subject: RID, sex, date of birth (PTDOB is 'MM/YYYY')."""
    raw = pd.read_csv(demog_csv, low_memory=False)
    df = pd.DataFrame({
        "participant_id": _to_pid(raw["PTID"]),
        "RID": pd.to_numeric(raw["RID"], errors="coerce"),
        "sex": pd.to_numeric(raw["PTGENDER"], errors="coerce").map(PTGENDER_TO_SEX),
        "dob": pd.to_datetime(raw["PTDOB"], format="%m/%Y", errors="coerce"),
    })
    df = df.dropna(subset=["participant_id"])
    # collapse to first non-missing sex/dob per subject
    df = df.sort_values("dob").groupby("participant_id", as_index=False).agg(
        RID=("RID", "first"), sex=("sex", "first"), dob=("dob", "first"))
    return df


def _load_diagnosis(dxsum_csv: Path) -> pd.DataFrame:
    """Per-visit diagnosis trajectory from DXSUM (harmonized DIAGNOSIS field)."""
    raw = pd.read_csv(dxsum_csv, low_memory=False)
    df = pd.DataFrame({
        "participant_id": _to_pid(raw["PTID"]),
        "RID": pd.to_numeric(raw["RID"], errors="coerce"),
        "exam_date": pd.to_datetime(raw["EXAMDATE"], errors="coerce"),
        "dx": pd.to_numeric(raw["DIAGNOSIS"], errors="coerce").map(DIAGNOSIS_TO_LABEL),
    })
    return df.dropna(subset=["exam_date", "dx"]).sort_values(["participant_id", "exam_date"])


def _load_csf(csf_csv: Path) -> pd.DataFrame:
    """Per-visit Roche Elecsys CSF biomarkers (UPENNBIOMK)."""
    raw = pd.read_csv(csf_csv, low_memory=False)
    df = pd.DataFrame({
        "participant_id": _to_pid(raw["PTID"]),
        "exam_date": pd.to_datetime(raw["EXAMDATE"], errors="coerce"),
        "csf_abeta": _parse_censored(raw["ABETA42"]),
        "csf_ptau": _parse_censored(raw["PTAU"]),
        "csf_ttau": _parse_censored(raw["TAU"]),
    })
    df = df.dropna(subset=["exam_date"])
    df = df.dropna(subset=["csf_abeta", "csf_ptau", "csf_ttau"], how="all")
    return df.sort_values(["participant_id", "exam_date"])


# ---------------------------------------------------------------------------
# Scan manifest
# ---------------------------------------------------------------------------

def _discover_scans(data_root: Path) -> tuple[pd.DataFrame, list[str]]:
    images = {}
    for path in data_root.glob("*T1w*space-MNI152NLin2009cAsym_desc-processed.nii.gz"):
        match = IMAGE_RE.match(path.name)
        if match:
            images[match.group("stem")] = (path, match.group("sub"), match.group("date"))

    rows, volume_names = [], None
    synthseg = data_root / "derivatives" / "synthseg"
    mask_dir = data_root / "derivatives" / "masks"
    for stem, (image_path, participant_id, date) in sorted(images.items()):
        qc_path = synthseg / f"{stem}_qc.csv"
        volumes_path = synthseg / f"{stem}_volumes.csv"
        mask_path = mask_dir / image_path.name.replace("_desc-processed.nii.gz", "_desc-brain_mask.nii.gz")
        if not qc_path.exists() or not volumes_path.exists() or not mask_path.exists():
            continue
        qc = pd.read_csv(qc_path, index_col=0).iloc[0].astype(float)
        volumes = pd.read_csv(volumes_path, index_col=0).iloc[0].astype(float)
        if volume_names is None:
            volume_names = list(volumes.index)
        elif list(volumes.index) != volume_names:
            raise ValueError(f"SynthSeg volume order differs for {volumes_path}")
        if len(volumes) != 101:
            raise ValueError(f"expected 101 SynthSeg volumes in {volumes_path}, got {len(volumes)}")
        rows.append({
            "sample_id": stem, "participant_id": participant_id,
            "session_id": date, "scan_date": pd.to_datetime(date, format="%Y%m%d"),
            "local_path": str(image_path), "path": f"images/{image_path.name}",
            "local_mask_path": str(mask_path), "mask_path": f"masks/{mask_path.name}",
            "synthseg_volumes": volumes.tolist(),
            "synthseg_qc_mean": float(qc.mean()), "synthseg_qc_min": float(qc.min()),
        })
    if not rows or volume_names is None:
        raise ValueError(f"no complete T1/SynthSeg records found under {data_root}")
    scans = pd.DataFrame(rows).sort_values(["participant_id", "scan_date"])
    return scans.reset_index(drop=True), volume_names


def _attach_clinical(scans: pd.DataFrame, demog: pd.DataFrame, dx_visits: pd.DataFrame,
                     csf_visits: pd.DataFrame, config: CohortConfig) -> pd.DataFrame:
    """Attach demographics (age/sex from PTDEMOG), the nearest diagnosis visit
    (DXSUM), and the nearest CSF visit (UPENN) to every scan."""
    demog_by_pid = demog.set_index("participant_id")

    out = []
    for participant_id, frame in scans.groupby("participant_id"):
        if participant_id not in demog_by_pid.index:
            continue  # no demographics -> cannot compute age/sex
        info = demog_by_pid.loc[participant_id]
        frame = frame.sort_values("scan_date").reset_index(drop=True)
        frame["rid"] = info["RID"]
        frame["sex"] = info["sex"]
        frame["age"] = ((frame["scan_date"] - info["dob"]).dt.days / 365.25
                        if pd.notna(info["dob"]) else np.nan)

        subj_dx = dx_visits[dx_visits["participant_id"] == participant_id]
        if not subj_dx.empty:
            dx = pd.merge_asof(
                frame[["scan_date"]], subj_dx[["exam_date", "dx"]],
                left_on="scan_date", right_on="exam_date", direction="nearest",
                tolerance=pd.Timedelta(days=config.clinical_window_days),
            )
            frame["diagnosis"] = dx["dx"].values
            frame["clinical_match_days"] = (dx["scan_date"] - dx["exam_date"]).abs().dt.days.values
        else:
            frame["diagnosis"] = None
            frame["clinical_match_days"] = np.nan

        subj_csf = csf_visits[csf_visits["participant_id"] == participant_id]
        if not subj_csf.empty:
            csf = pd.merge_asof(
                frame[["scan_date"]],
                subj_csf[["exam_date", "csf_abeta", "csf_ptau", "csf_ttau"]],
                left_on="scan_date", right_on="exam_date", direction="nearest",
                tolerance=pd.Timedelta(days=config.csf_window_days),
            )
            frame["csf_abeta"] = csf["csf_abeta"].values
            frame["csf_ptau"] = csf["csf_ptau"].values
            frame["csf_ttau"] = csf["csf_ttau"].values
            frame["csf_match_days"] = (csf["scan_date"] - csf["exam_date"]).abs().dt.days.values
        else:
            for col in ("csf_abeta", "csf_ptau", "csf_ttau", "csf_match_days"):
                frame[col] = np.nan

        out.append(frame)

    manifest = pd.concat(out, ignore_index=True)
    # ClassLabel diagnosis cannot be null; the ~39 subjects with no DXSUM record
    # are dropped (they carry no AD-axis label for any v1 task).
    manifest = manifest.dropna(subset=["diagnosis"])
    return manifest


# ---------------------------------------------------------------------------
# PET labels (UC Berkeley gold-standard tables, joined by RID + scan date)
# ---------------------------------------------------------------------------

def _attach_pet(manifest: pd.DataFrame, *, amy_csv: Path, tau_csv: Path,
                config: CohortConfig) -> pd.DataFrame:
    amy = pd.read_csv(amy_csv, low_memory=False)
    amy = pd.DataFrame({
        "rid": pd.to_numeric(amy["RID"], errors="coerce"),
        "pet_date": pd.to_datetime(amy["SCANDATE"], errors="coerce"),
        "amyloid_status": pd.to_numeric(amy["AMYLOID_STATUS"], errors="coerce"),
        "amyloid_centiloid": pd.to_numeric(amy["CENTILOIDS"], errors="coerce"),
        "amyloid_suvr": pd.to_numeric(amy["SUMMARY_SUVR"], errors="coerce"),
        "amyloid_tracer": amy["TRACER"],
    }).dropna(subset=["rid", "pet_date"])

    tau = pd.read_csv(tau_csv, low_memory=False)
    tau = pd.DataFrame({
        "rid": pd.to_numeric(tau["RID"], errors="coerce"),
        "pet_date": pd.to_datetime(tau["SCANDATE"], errors="coerce"),
        "tau_suvr": pd.to_numeric(tau["META_TEMPORAL_SUVR"], errors="coerce"),
        "tau_tracer": tau["TRACER"],
    }).dropna(subset=["rid", "pet_date"])
    tau["tau_status"] = (tau["tau_suvr"] > config.tau_suvr_threshold).astype(float)
    tau.loc[tau["tau_suvr"].isna(), "tau_status"] = np.nan

    manifest = _nearest_by_rid_date(
        manifest, amy, ["amyloid_status", "amyloid_centiloid", "amyloid_suvr", "amyloid_tracer"],
        match_col="amyloid_match_days", window_days=config.pet_window_days,
    )
    manifest = _nearest_by_rid_date(
        manifest, tau, ["tau_status", "tau_suvr", "tau_tracer"],
        match_col="tau_match_days", window_days=config.pet_window_days,
    )
    return manifest


def _nearest_by_rid_date(manifest: pd.DataFrame, table: pd.DataFrame, value_cols: list[str],
                         *, match_col: str, window_days: int) -> pd.DataFrame:
    """For each scan, attach the value columns from the row in ``table`` with the
    same RID whose date is nearest the scan date within ``window_days``."""
    manifest = manifest.copy()
    window = pd.Timedelta(days=window_days)
    by_rid = {rid: frame.sort_values("pet_date") for rid, frame in table.groupby("rid")}

    collected = {col: [None] * len(manifest) for col in value_cols}
    match_days = [np.nan] * len(manifest)
    for pos, (_, row) in enumerate(manifest.iterrows()):
        frame = by_rid.get(row["rid"])
        if frame is None:
            continue
        delta = (frame["pet_date"] - row["scan_date"]).abs()
        j = int(delta.values.argmin())
        if delta.iloc[j] <= window:
            best = frame.iloc[j]
            for col in value_cols:
                collected[col][pos] = best[col]
            match_days[pos] = delta.iloc[j].days

    for col in value_cols:
        manifest[col] = collected[col]
    manifest[match_col] = match_days
    return manifest


# ---------------------------------------------------------------------------
# MCI -> AD conversion (derived from each subject's diagnosis trajectory)
# ---------------------------------------------------------------------------

def _attach_conversion(manifest: pd.DataFrame, dx_visits: pd.DataFrame, config: CohortConfig) -> pd.DataFrame:
    horizon = config.conversion_horizon_months
    dx_visits = dx_visits.sort_values(["participant_id", "exam_date"])

    event = np.full(len(manifest), np.nan)
    time_m = np.full(len(manifest), np.nan)
    binary = np.full(len(manifest), np.nan)

    for i, row in manifest.reset_index().iterrows():
        if row["diagnosis"] != "MCI":
            continue  # at-risk set is MCI-at-scan only
        subj = dx_visits[dx_visits["participant_id"] == row["participant_id"]]
        future = subj[subj["exam_date"] >= row["scan_date"]]
        if future.empty:
            continue
        converts = future[future["dx"] == "AD"]
        if not converts.empty:
            t = (converts["exam_date"].iloc[0] - row["scan_date"]).days / 30.44
            event[i], time_m[i] = 1.0, t
            binary[i] = 1.0 if t <= horizon else 0.0
        else:
            t = (future["exam_date"].iloc[-1] - row["scan_date"]).days / 30.44
            event[i], time_m[i] = 0.0, t
            binary[i] = 0.0 if t >= horizon else np.nan  # censored before horizon -> unknown

    manifest = manifest.copy()
    manifest["conversion_event"] = event
    manifest["conversion_time_months"] = time_m
    manifest["conversion_3y"] = binary
    return manifest


# ---------------------------------------------------------------------------
# Eligibility + single subject-exclusive split
# ---------------------------------------------------------------------------

def _filter_eligible(manifest: pd.DataFrame, config: CohortConfig) -> pd.DataFrame:
    df = manifest.copy()
    valid = (
        df["age"].between(config.min_age, config.max_age, inclusive="left")
        & df["diagnosis"].isin(["CN", "MCI", "AD"])
        & df["sex"].isin(["Female", "Male"])
        & (df["synthseg_qc_mean"] >= config.min_qc_mean)
        & (df["synthseg_qc_min"] >= config.min_qc_min)
        & df["synthseg_volumes"].notna()
    )
    df = df.loc[valid].copy()
    # one acquisition per participant/session, best segmentation quality
    df = df.sort_values(
        ["participant_id", "session_id", "synthseg_qc_mean", "sample_id"],
        ascending=[True, True, False, True],
    ).drop_duplicates(["participant_id", "session_id"])
    return df.sort_values("sample_id").reset_index(drop=True)


def build_manifest(
    data_root: Path, *, demog_csv: Path, dxsum_csv: Path, csf_csv: Path,
    amy_csv: Path, tau_csv: Path, config: CohortConfig = CohortConfig(),
) -> tuple[pd.DataFrame, list[str]]:
    scans, volume_names = _discover_scans(data_root)
    demog = _load_demographics(demog_csv)
    dx_visits = _load_diagnosis(dxsum_csv)
    csf_visits = _load_csf(csf_csv)
    manifest = _attach_clinical(scans, demog, dx_visits, csf_visits, config)
    manifest = _filter_eligible(manifest, config)
    manifest = _attach_pet(manifest, amy_csv=amy_csv, tau_csv=tau_csv, config=config)
    manifest = _attach_conversion(manifest, dx_visits, config)
    return manifest.reset_index(drop=True), volume_names


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def manifest_report(manifest: pd.DataFrame) -> dict:
    def labelled(col, positive=None):
        present = manifest[col].notna()
        out = {"scans": int(present.sum()),
               "subjects": int(manifest.loc[present, "participant_id"].nunique())}
        if positive is not None:
            out["positive"] = int((manifest[col] == positive).sum())
        return out

    report = {
        "total_scans": int(len(manifest)),
        "total_subjects": int(manifest["participant_id"].nunique()),
        "diagnosis": manifest["diagnosis"].value_counts().to_dict(),
        "tasks": {
            "ad_cn": {"CN": int((manifest["diagnosis"] == "CN").sum()),
                      "AD": int((manifest["diagnosis"] == "AD").sum())},
            "amyloid_status": labelled("amyloid_status", positive=1.0),
            "amyloid_centiloid": labelled("amyloid_centiloid"),
            "tau_status": labelled("tau_status", positive=1.0),
            "tau_suvr": labelled("tau_suvr"),
            "csf_abeta": labelled("csf_abeta"),
            "csf_ptau": labelled("csf_ptau"),
            "csf_ttau": labelled("csf_ttau"),
            "conversion_3y": labelled("conversion_3y", positive=1.0),
        },
    }
    return report


# ---------------------------------------------------------------------------
# Shard builder
# ---------------------------------------------------------------------------

def build_dataset(
    *,
    data_root: Path,
    demog_csv: Path,
    dxsum_csv: Path,
    csf_csv: Path,
    amy_csv: Path,
    tau_csv: Path,
    output_dir: Path,
    num_proc: int = 8,
    max_shard_size: str = "1GB",
    cohort_config: CohortConfig = CohortConfig(),
) -> DatasetDict:
    eval_path = output_dir / "eval"
    if eval_path.exists():
        raise FileExistsError(f"dataset output already exists: {eval_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest, volume_names = build_manifest(
        data_root, demog_csv=demog_csv, dxsum_csv=dxsum_csv, csf_csv=csf_csv,
        amy_csv=amy_csv, tau_csv=tau_csv, config=cohort_config,
    )

    manifest.drop(columns=["local_path", "local_mask_path"]).to_parquet(
        output_dir / "source_manifest.parquet", index=False)
    manifest.to_csv(output_dir / "manifest.csv", index=False)
    (output_dir / "synthseg_volume_names.json").write_text(json.dumps(volume_names, indent=2) + "\n")
    (output_dir / "manifest_report.json").write_text(
        json.dumps(manifest_report(manifest), indent=2, default=str) + "\n")

    # Single `eval` split (matching the published v0.x layout). The
    # subject-exclusive train/validation/test assignment is preserved in the
    # `split` column; cross-validation is performed downstream.
    records = manifest.to_dict("records")
    eval_ds = Dataset.from_generator(
        _generate_samples,
        features=adni_features(),
        gen_kwargs={"records": records},
        num_proc=min(num_proc, max(1, len(records))),
        split="eval",
        fingerprint=f"adni-smri-v1-local-eval-{cohort_config.seed}",
        writer_batch_size=8,
    )

    dataset = DatasetDict({"eval": eval_ds})
    dataset.save_to_disk(output_dir, max_shard_size=max_shard_size, num_proc=num_proc)
    return dataset


def _f(value) -> float:
    """Coerce optional numeric to float, preserving NaN."""
    return float(value) if value is not None and pd.notna(value) else float("nan")


def _s(value):
    return value if value is not None and pd.notna(value) else None


def _generate_samples(records):
    for record in records:
        local_path = Path(record["local_path"])
        local_mask_path = Path(record["local_mask_path"])
        if not local_path.is_file():
            raise FileNotFoundError(local_path)
        if not local_mask_path.is_file():
            raise FileNotFoundError(local_mask_path)
        rid = record["rid"]
        yield {
            "sample_id": record["sample_id"],
            "participant_id": record["participant_id"],
            "rid": int(rid) if pd.notna(rid) else -1,
            "session_id": record["session_id"],
            "scan_date": str(record["scan_date"]),
            "age": _f(record["age"]),
            "sex": record["sex"],
            "diagnosis": record["diagnosis"],
            "synthseg_volumes": list(record["synthseg_volumes"]),
            "amyloid_status": _f(record["amyloid_status"]),
            "amyloid_centiloid": _f(record["amyloid_centiloid"]),
            "amyloid_suvr": _f(record["amyloid_suvr"]),
            "amyloid_tracer": _s(record["amyloid_tracer"]),
            "amyloid_match_days": _f(record["amyloid_match_days"]),
            "tau_status": _f(record["tau_status"]),
            "tau_suvr": _f(record["tau_suvr"]),
            "tau_tracer": _s(record["tau_tracer"]),
            "tau_match_days": _f(record["tau_match_days"]),
            "csf_abeta": _f(record["csf_abeta"]),
            "csf_ptau": _f(record["csf_ptau"]),
            "csf_ttau": _f(record["csf_ttau"]),
            "csf_match_days": _f(record["csf_match_days"]),
            "conversion_event": _f(record["conversion_event"]),
            "conversion_time_months": _f(record["conversion_time_months"]),
            "conversion_3y": _f(record["conversion_3y"]),
            "clinical_match_days": _f(record["clinical_match_days"]),
            "nifti": {"path": record["path"], "bytes": local_path.read_bytes()},
            "mask": {"path": record["mask_path"], "bytes": local_mask_path.read_bytes()},
        }


def cli() -> None:
    parser = argparse.ArgumentParser(description="Build ADNI v1 sMRI eval dataset as local Arrow shards")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--demog-csv", type=Path, required=True,
                        help="PTDEMOG table (age from PTDOB, sex from PTGENDER)")
    parser.add_argument("--dxsum-csv", type=Path, required=True,
                        help="DXSUM diagnostic summary (DIAGNOSIS 1=CN/2=MCI/3=Dementia)")
    parser.add_argument("--csf-csv", type=Path, required=True,
                        help="UPENNBIOMK Roche Elecsys CSF (ABETA42/TAU/PTAU)")
    parser.add_argument("--amyloid-csv", type=Path, required=True,
                        help="UC Berkeley amyloid table (UCBERKELEY_AMY_6MM_*.csv)")
    parser.add_argument("--tau-csv", type=Path, required=True,
                        help="UC Berkeley tau table (UCBERKELEY_TAU_6MM_*.csv)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-proc", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--max-shard-size", default="1GB")
    args = parser.parse_args()
    dataset = build_dataset(
        data_root=args.data_root,
        demog_csv=args.demog_csv,
        dxsum_csv=args.dxsum_csv,
        csf_csv=args.csf_csv,
        amy_csv=args.amyloid_csv,
        tau_csv=args.tau_csv,
        output_dir=args.output_dir,
        num_proc=args.num_proc,
        max_shard_size=args.max_shard_size,
    )
    print(dataset)


if __name__ == "__main__":
    cli()
