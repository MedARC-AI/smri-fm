"""Rebuild adni-smri-v01-hub with brain_mask added.

Reads the existing cohort.parquet (which has local_path for each scan),
derives the local mask path, and rebuilds the dataset using Dataset.from_generator.
The existing cohort split / train_rank assignments are preserved exactly.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from datasets import ClassLabel, Dataset, DatasetDict, Features, Nifti, Sequence, Value


def adni_features_with_mask() -> Features:
    return Features({
        "sample_id": Value("string"),
        "participant_id": Value("string"),
        "session_id": Value("string"),
        "scan_date": Value("string"),
        "age": Value("float32"),
        "sex": ClassLabel(names=["Female", "Male"]),
        "diagnosis": ClassLabel(names=["CN", "AD"]),
        "synthseg_volumes": Sequence(Value("float32"), length=101),
        "train_rank": Value("int32"),
        "nifti": Nifti(),
        "mask": Nifti(),
    })


def _generate_samples(records, mask_dir: Path):
    for record in records:
        local_path = Path(record["local_path"])
        mask_basename = local_path.name.replace(
            "_desc-processed.nii.gz", "_desc-brain_mask.nii.gz"
        )
        local_mask_path = mask_dir / mask_basename
        mask_path = f"masks/{mask_basename}"

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
            "mask": {"path": mask_path, "bytes": local_mask_path.read_bytes()},
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild ADNI hub with brain_mask column")
    parser.add_argument("--cohort", type=Path, required=True,
                        help="Path to cohort.parquet")
    parser.add_argument("--mask-dir", type=Path, required=True,
                        help="Directory containing brain mask .nii.gz files")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output directory for the new hub")
    parser.add_argument("--num-proc", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--max-shard-size", default="1GB")
    args = parser.parse_args()

    cohort = pd.read_parquet(args.cohort)
    print(f"Cohort: {len(cohort)} rows, splits: {cohort['split'].value_counts().to_dict()}")

    # Validate all masks exist before starting
    mask_dir = args.mask_dir.resolve()
    missing = []
    for lp in cohort["local_path"]:
        mask_basename = Path(lp).name.replace(
            "_desc-processed.nii.gz", "_desc-brain_mask.nii.gz"
        )
        if not (mask_dir / mask_basename).exists():
            missing.append(mask_basename)
    if missing:
        raise FileNotFoundError(f"{len(missing)} masks missing, e.g. {missing[:3]}")
    print(f"All {len(cohort)} masks verified.")

    output_dir = args.output.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}. Remove it first.")
    output_dir.mkdir(parents=True)

    features = adni_features_with_mask()
    datasets = {}
    for split in ["train", "validation", "test"]:
        records = cohort.loc[cohort["split"] == split].to_dict("records")
        print(f"Building {split}: {len(records)} records...")
        datasets[split] = Dataset.from_generator(
            _generate_samples,
            features=features,
            gen_kwargs={"records": records, "mask_dir": mask_dir},
            num_proc=min(args.num_proc, len(records)),
            split=split,
            writer_batch_size=8,
        )
        print(f"  {split} done: {datasets[split]}")

    dataset = DatasetDict(datasets)
    print("Saving to disk...")
    dataset.save_to_disk(output_dir, max_shard_size=args.max_shard_size, num_proc=args.num_proc)
    print(f"Done → {output_dir}")
    print(dataset)


if __name__ == "__main__":
    main()
