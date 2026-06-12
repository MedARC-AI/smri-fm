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
from sklearn.model_selection import train_test_split


def adni_features() -> Features:
    return Features({
        "sample_id": Value("string"), "participant_id": Value("string"),
        "session_id": Value("string"), "scan_date": Value("string"),
        "age": Value("float32"), "sex": ClassLabel(names=["Female", "Male"]),
        "diagnosis": ClassLabel(names=["CN", "AD"]),
        "synthseg_volumes": Sequence(Value("float32"), length=101),
        "train_rank": Value("int32"),
        "nifti": Nifti(), "mask": Nifti(),
    })


IMAGE_RE = re.compile(r"^(?P<stem>sub-(?P<sub>[^_]+)_ses-(?P<date>\d{8})_T1w(?:_[^_]+)?)_space-")
AGE_EDGES = [60, 65, 70, 75, 80, 85, 90]
AGE_LABELS = ["60-64", "65-69", "70-74", "75-79", "80-84", "85-89"]


@dataclass(frozen=True)
class CohortConfig:
    train_size: int = 2000
    validation_size: int = 500
    test_size: int = 500
    seed: int = 4466
    min_age: float = 60.0
    max_age: float = 90.0
    min_qc_mean: float = 0.80
    min_qc_min: float = 0.60
    soft_session_limit: int = 3
    split_trials: int = 500


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def build_manifest(data_root: Path, clinical_csv: Path) -> tuple[pd.DataFrame, list[str]]:
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
    clinical = pd.read_csv(clinical_csv, low_memory=False)[
        ["PTID", "EXAMDATE", "AGE", "Years_bl", "PTGENDER", "DX"]
    ].copy()
    clinical["participant_id"] = clinical["PTID"].str.replace("_", "", regex=False)
    clinical["exam_date"] = pd.to_datetime(clinical["EXAMDATE"])
    clinical["age"] = (pd.to_numeric(clinical["AGE"], errors="coerce") +
                       pd.to_numeric(clinical["Years_bl"], errors="coerce").fillna(0))

    joined = []
    for participant_id, frame in scans.groupby("participant_id"):
        subject_clinical = clinical[clinical["participant_id"] == participant_id].sort_values("exam_date")
        if subject_clinical.empty:
            continue
        frame = frame.drop(columns="participant_id").sort_values("scan_date")
        matched = pd.merge_asof(
            frame, subject_clinical, left_on="scan_date", right_on="exam_date",
            direction="nearest", tolerance=pd.Timedelta(days=90),
        )
        matched["participant_id"] = participant_id
        matched["clinical_match_days"] = (matched["scan_date"] - matched["exam_date"]).abs().dt.days
        joined.append(matched)

    manifest = pd.concat(joined, ignore_index=True)
    manifest = manifest.dropna(subset=["age", "PTGENDER", "DX"])
    manifest = manifest.rename(columns={"PTGENDER": "sex", "DX": "diagnosis"})
    # One acquisition per participant/session, selected by segmentation quality.
    manifest = manifest.sort_values(
        ["participant_id", "session_id", "synthseg_qc_mean", "sample_id"],
        ascending=[True, True, False, True],
    ).drop_duplicates(["participant_id", "session_id"])
    columns = [
        "sample_id", "participant_id", "session_id", "scan_date", "age", "sex",
        "diagnosis", "synthseg_volumes", "synthseg_qc_mean", "synthseg_qc_min",
        "clinical_match_days", "path", "local_path", "mask_path", "local_mask_path",
    ]
    return manifest[columns].reset_index(drop=True), volume_names


# ---------------------------------------------------------------------------
# Cohort
# ---------------------------------------------------------------------------

def build_cohort(manifest: pd.DataFrame, config: CohortConfig = CohortConfig()) -> pd.DataFrame:
    pool = _prepare_eligible_pool(manifest, config)
    subjects = _make_subject_table(pool)
    sizes = {"train": config.train_size, "validation": config.validation_size, "test": config.test_size}
    best = None
    for trial in range(config.split_trials):
        seed = config.seed + trial
        train_subs, held_subs = train_test_split(
            subjects, test_size=0.30, random_state=seed, stratify=subjects["stratum"]
        )
        val_subs, test_subs = train_test_split(
            held_subs, test_size=0.50, random_state=seed + 10_000,
            stratify=held_subs["diagnosis"] + "|" + held_subs["sex"],
        )
        subject_splits = {"train": train_subs, "validation": val_subs, "test": test_subs}
        selected, score, feasible = {}, 0.0, True
        for split, split_subjects in subject_splits.items():
            split_pool = pool[pool["participant_id"].isin(split_subjects["participant_id"])]
            try:
                rows = _select_split(
                    split_pool, size=sizes[split],
                    seed=seed + {"train": 0, "validation": 100, "test": 200}[split],
                    soft_limit=config.soft_session_limit,
                )
            except ValueError:
                feasible = False
                break
            selected[split] = rows
            score += _balance_score(rows)
        if feasible and (best is None or score < best[0]):
            best = score, selected
    if best is None:
        raise ValueError("unable to construct a subject-exclusive cohort with requested quotas")

    frames = []
    for split, rows in best[1].items():
        rows = rows.copy()
        rows["split"] = split
        rows["train_rank"] = -1
        if split == "train":
            rows = _assign_train_rank(rows, config.seed)
        frames.append(rows)
    cohort = pd.concat(frames, ignore_index=True)
    _validate_cohort(cohort, config)
    return cohort.sort_values(["split", "train_rank", "sample_id"]).reset_index(drop=True)


def cohort_report(cohort: pd.DataFrame, soft_limit: int = 3) -> dict:
    report = {}
    for split, frame in cohort.groupby("split", sort=False):
        per_subject = frame.groupby("participant_id").size()
        report[split] = {
            "sessions": int(len(frame)), "subjects": int(frame["participant_id"].nunique()),
            "diagnosis": frame["diagnosis"].value_counts().to_dict(),
            "sex": frame["sex"].value_counts().to_dict(),
            "age": {k: float(v) for k, v in {
                "mean": frame["age"].mean(), "std": frame["age"].std(),
                "min": frame["age"].min(), "max": frame["age"].max(),
            }.items()},
            "age_percent": {str(k): float(v) for k, v in
                (frame["age_bin"].value_counts(normalize=True).sort_index() * 100).items()},
            "subjects_over_soft_limit": int((per_subject > soft_limit).sum()),
            "sessions_beyond_soft_limit": int((per_subject - soft_limit).clip(lower=0).sum()),
            "max_sessions_per_subject": int(per_subject.max()),
        }
    return report


def _prepare_eligible_pool(manifest: pd.DataFrame, config: CohortConfig) -> pd.DataFrame:
    df = manifest.copy()
    df["diagnosis"] = df["diagnosis"].replace({"Dementia": "AD"})
    valid = (
        df["age"].between(config.min_age, config.max_age, inclusive="left")
        & df["diagnosis"].isin(["CN", "AD"])
        & df["sex"].isin(["Female", "Male"])
        & (df["synthseg_qc_mean"] >= config.min_qc_mean)
        & (df["synthseg_qc_min"] >= config.min_qc_min)
        & df["synthseg_volumes"].notna()
    )
    df = df.loc[valid].copy()
    df["scan_date"] = pd.to_datetime(df["scan_date"])
    df["age_bin"] = pd.cut(df["age"], AGE_EDGES, right=False, labels=AGE_LABELS)
    return df.sort_values("sample_id").reset_index(drop=True)


def _validate_cohort(cohort: pd.DataFrame, config: CohortConfig) -> None:
    expected = {"train": config.train_size, "validation": config.validation_size, "test": config.test_size}
    if cohort["split"].value_counts().to_dict() != expected:
        raise ValueError("cohort split counts do not match configuration")
    for split, size in expected.items():
        counts = cohort.loc[cohort["split"] == split, "diagnosis"].value_counts()
        if counts.get("CN", 0) != size // 2 or counts.get("AD", 0) != size // 2:
            raise ValueError(f"{split} is not diagnosis-balanced")
    if (cohort.groupby("participant_id")["split"].nunique() != 1).any():
        raise ValueError("participants overlap between splits")
    ranks = cohort.loc[cohort["split"] == "train", "train_rank"].sort_values().to_numpy()
    if not np.array_equal(ranks, np.arange(config.train_size)):
        raise ValueError("train_rank must be contiguous from zero")


def _make_subject_table(pool: pd.DataFrame) -> pd.DataFrame:
    def mode(s):
        return s.value_counts().index[0]
    subjects = pool.groupby("participant_id").agg(
        diagnosis=("diagnosis", mode), sex=("sex", mode), age=("age", "median")
    ).reset_index()
    subjects["age_bin"] = pd.cut(subjects["age"], AGE_EDGES, right=False, labels=AGE_LABELS)
    subjects["stratum"] = subjects["diagnosis"] + "|" + subjects["sex"] + "|" + subjects["age_bin"].astype(str)
    counts = subjects["stratum"].value_counts()
    rare = subjects["stratum"].map(counts) < 4
    subjects.loc[rare, "stratum"] = subjects.loc[rare, "diagnosis"] + "|" + subjects.loc[rare, "sex"]
    return subjects


def _select_split(pool: pd.DataFrame, *, size: int, seed: int, soft_limit: int) -> pd.DataFrame:
    selected = []
    for dx_index, diagnosis in enumerate(["CN", "AD"]):
        candidates = pool[pool["diagnosis"] == diagnosis].copy()
        target = size // 2
        if len(candidates) < target:
            raise ValueError(f"not enough {diagnosis} sessions")
        counts = candidates.groupby(["sex", "age_bin"], observed=True).size()
        candidates["balance_weight"] = [
            1.0 / counts.loc[(row.sex, row.age_bin)] for row in candidates.itertuples()
        ]
        median_date = candidates.groupby("participant_id")["scan_date"].transform("median")
        candidates["date_distance"] = (candidates["scan_date"] - median_date).abs()
        rng = np.random.default_rng(seed + dx_index)
        candidates["tie_breaker"] = rng.random(len(candidates))
        candidates = candidates.sort_values(["participant_id", "date_distance", "tie_breaker"])
        candidates["subject_rank"] = candidates.groupby("participant_id").cumcount() + 1
        primary = candidates[candidates["subject_rank"] <= soft_limit]
        overflow = candidates[candidates["subject_rank"] > soft_limit]
        chosen = _weighted_sample(primary, min(target, len(primary)), rng)
        if len(chosen) < target:
            chosen = pd.concat([chosen, _weighted_sample(overflow, target - len(chosen), rng)])
        selected.append(chosen)
    return pd.concat(selected).drop(columns=["balance_weight", "date_distance", "tie_breaker"])


def _weighted_sample(frame: pd.DataFrame, size: int, rng: np.random.Generator) -> pd.DataFrame:
    if size == 0:
        return frame.iloc[:0]
    if len(frame) < size:
        raise ValueError("insufficient rows for weighted sample")
    weights = frame["balance_weight"].to_numpy(dtype=float)
    weights /= weights.sum()
    return frame.iloc[rng.choice(len(frame), size=size, replace=False, p=weights)]


def _balance_score(frame: pd.DataFrame) -> float:
    sex_error = abs((frame["sex"] == "Female").mean() - 0.5)
    fractions = frame["age_bin"].value_counts(normalize=True)
    return float(sex_error + (fractions - fractions.mean()).abs().mean())


def _assign_train_rank(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    ordered = []
    for dx_index, diagnosis in enumerate(["AD", "CN"]):
        group = frame[frame["diagnosis"] == diagnosis].copy()
        group["tie_breaker"] = np.random.default_rng(seed + dx_index).random(len(group))
        group = group.sort_values(["subject_rank", "tie_breaker"])
        group["stratum_rank"] = group.groupby(
            ["subject_rank", "age_bin", "sex"], observed=True
        ).cumcount()
        group = group.sort_values(["subject_rank", "stratum_rank", "age_bin", "sex", "tie_breaker"])
        group["diagnosis_rank"] = np.arange(len(group))
        ordered.append(group)
    ranked = pd.concat(ordered).sort_values(["diagnosis_rank", "diagnosis"])
    ranked["train_rank"] = np.arange(len(ranked))
    return ranked.drop(columns=["tie_breaker", "stratum_rank", "diagnosis_rank"])


# ---------------------------------------------------------------------------
# Shard builder
# ---------------------------------------------------------------------------

def build_dataset(
    *,
    data_root: Path,
    clinical_csv: Path,
    output_dir: Path,
    num_proc: int = 8,
    max_shard_size: str = "1GB",
    cohort_config: CohortConfig = CohortConfig(),
) -> DatasetDict:
    dataset_path = output_dir / "dataset"
    if dataset_path.exists():
        raise FileExistsError(f"dataset output already exists: {dataset_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest, volume_names = build_manifest(data_root, clinical_csv)
    cohort = build_cohort(manifest, cohort_config)

    manifest.drop(columns=["local_path", "local_mask_path"]).to_parquet(output_dir / "source_manifest.parquet", index=False)
    cohort.to_parquet(output_dir / "cohort.parquet", index=False)
    cohort.to_csv(output_dir / "cohort.csv", index=False)
    (output_dir / "synthseg_volume_names.json").write_text(
        json.dumps(volume_names, indent=2) + "\n"
    )
    (output_dir / "cohort_report.json").write_text(
        json.dumps(cohort_report(cohort, cohort_config.soft_session_limit), indent=2) + "\n"
    )

    datasets = {}
    for split in ["train", "validation", "test"]:
        records = cohort.loc[cohort["split"] == split].to_dict("records")
        datasets[split] = Dataset.from_generator(
            _generate_samples,
            features=adni_features(),
            gen_kwargs={"records": records},
            num_proc=min(num_proc, len(records)),
            split=split,
            fingerprint=f"adni-smri-v01-local-{split}-{cohort_config.seed}",
            writer_batch_size=8,
        )

    dataset = DatasetDict(datasets)
    dataset.save_to_disk(dataset_path, max_shard_size=max_shard_size, num_proc=num_proc)
    return dataset


def _generate_samples(records):
    for record in records:
        local_path = Path(record["local_path"])
        local_mask_path = Path(record["local_mask_path"])
        if not local_path.is_file():
            raise FileNotFoundError(local_path)
        if not local_mask_path.is_file():
            raise FileNotFoundError(local_mask_path)
        yield {
            "sample_id": record["sample_id"],
            "participant_id": record["participant_id"],
            "session_id": record["session_id"],
            "scan_date": str(record["scan_date"]),
            "age": float(record["age"]),
            "sex": record["sex"],
            "diagnosis": record["diagnosis"],
            "synthseg_volumes": list(record["synthseg_volumes"]),
            "train_rank": int(record["train_rank"]),
            "nifti": {"path": record["path"], "bytes": local_path.read_bytes()},
            "mask": {"path": record["mask_path"], "bytes": local_mask_path.read_bytes()},
        }


def cli() -> None:
    parser = argparse.ArgumentParser(description="Build ADNI sMRI eval dataset as local Arrow shards")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--clinical-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-proc", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--max-shard-size", default="1GB")
    args = parser.parse_args()
    dataset = build_dataset(
        data_root=args.data_root,
        clinical_csv=args.clinical_csv,
        output_dir=args.output_dir,
        num_proc=args.num_proc,
        max_shard_size=args.max_shard_size,
    )
    print(dataset)


if __name__ == "__main__":
    cli()
