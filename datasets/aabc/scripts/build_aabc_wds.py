#!/usr/bin/env python
"""Build AABC structural MRI WebDataset shards."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import webdataset as wds
from tqdm import tqdm


DEFAULT_IMAGE_GLOB = (
    "../aabc/processed/*_space-MNI152NLin2009cAsym_desc-processed.nii.gz"
)
DEFAULT_MASK_DIR = "../aabc/processed/derivatives/masks"
DEFAULT_METADATA_CSV = (
    "/teamspace/studios/this_studio/metadata/AABC2_subjects_2026_04_30_10_04_46.csv"
)
DEFAULT_OUTPUT_PATTERN = "datasets/aabc/aabc-%06d.tar"
DEFAULT_MAXSIZE = 700 * 1024**2

PROCESSED_SUFFIX = "_space-MNI152NLin2009cAsym_desc-processed.nii.gz"
MASK_SUFFIX = "_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz"
SCAN_RE = re.compile(r"^(sub-[^_]+)_(ses-[^_]+)_([^_]+)$")

CORE_METADATA_FIELDS = (
    "age_open",
    "sex",
    "race",
    "ethnicity",
    "site",
    "scanner",
    "study",
    "days_from_V1",
    "yearquarter_event",
)


@dataclass(frozen=True)
class AABCRecord:
    key: str
    image_path: Path
    mask_path: Path
    subject: str
    subject_id: str
    session: str
    visit: str
    modality: str
    id_event: str
    metadata: dict[str, Any] | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-glob", default=DEFAULT_IMAGE_GLOB)
    parser.add_argument("--mask-dir", default=DEFAULT_MASK_DIR)
    parser.add_argument("--metadata-csv", default=DEFAULT_METADATA_CSV)
    parser.add_argument("--output-pattern", default=DEFAULT_OUTPUT_PATTERN)
    parser.add_argument("--maxsize", type=int, default=DEFAULT_MAXSIZE)
    parser.add_argument("--maxsize-mb", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--skip-shape-check", action="store_true")
    parser.add_argument("--allow-unmatched-metadata", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    maxsize = args.maxsize_mb * 1024**2 if args.maxsize_mb is not None else args.maxsize

    metadata = load_metadata(Path(args.metadata_csv))
    records = collect_records(
        image_glob=args.image_glob,
        mask_dir=Path(args.mask_dir),
        metadata=metadata,
        limit=args.limit,
        allow_unmatched_metadata=args.allow_unmatched_metadata,
    )

    validate_records(records, check_shapes=not args.skip_shape_check)
    report_records(records)

    if args.dry_run or args.validate_only:
        return

    output_pattern = Path(args.output_pattern)
    output_pattern.parent.mkdir(parents=True, exist_ok=True)
    assert_output_ready(str(output_pattern), overwrite=args.overwrite)

    with wds.ShardWriter(str(output_pattern), maxsize=maxsize) as sink:
        for record in tqdm(records, desc="writing shards", unit="scan"):
            sink.write(record_to_sample(record))


def load_metadata(path: Path) -> dict[str, dict[str, Any]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    metadata: dict[str, dict[str, Any]] = {}
    for row in rows:
        clean_row = {key: coerce_csv_value(value) for key, value in row.items()}
        id_event = clean_row.get("id_event")
        if not id_event:
            continue
        if id_event in metadata:
            raise ValueError(f"duplicate id_event in metadata CSV: {id_event}")
        metadata[str(id_event)] = clean_row
    return metadata


def collect_records(
    image_glob: str,
    mask_dir: Path,
    metadata: dict[str, dict[str, Any]],
    limit: int | None = None,
    allow_unmatched_metadata: bool = False,
) -> list[AABCRecord]:
    image_paths = [Path(path) for path in sorted(glob(image_glob))]
    if limit is not None:
        image_paths = image_paths[:limit]

    if not image_paths:
        raise FileNotFoundError(f"no processed images matched {image_glob!r}")

    records = []
    missing_masks = []
    unmatched_metadata = []

    for image_path in image_paths:
        key = scan_key(image_path)
        match = SCAN_RE.match(key)
        if match is None:
            raise ValueError(f"could not parse scan key from {image_path.name}: {key}")

        subject, session, modality = match.groups()
        subject_id = subject.removeprefix("sub-")
        visit = session.removeprefix("ses-")
        id_event = f"{subject_id}_{visit}"
        mask_path = mask_dir / f"{key}{MASK_SUFFIX}"

        if not mask_path.exists():
            missing_masks.append((image_path, mask_path))
            continue

        row = metadata.get(id_event)
        if row is None:
            unmatched_metadata.append(key)
            if not allow_unmatched_metadata:
                continue

        records.append(
            AABCRecord(
                key=key,
                image_path=image_path,
                mask_path=mask_path,
                subject=subject,
                subject_id=subject_id,
                session=session,
                visit=visit,
                modality=modality,
                id_event=id_event,
                metadata=row,
            )
        )

    if missing_masks:
        examples = "\n".join(f"  {image} -> {mask}" for image, mask in missing_masks[:10])
        raise FileNotFoundError(f"{len(missing_masks)} images are missing masks:\n{examples}")

    if unmatched_metadata and not allow_unmatched_metadata:
        examples = "\n".join(f"  {key}" for key in unmatched_metadata[:10])
        raise ValueError(
            f"{len(unmatched_metadata)} scans did not join to metadata; "
            f"use --allow-unmatched-metadata to shard them anyway.\n{examples}"
        )

    return records


def validate_records(records: list[AABCRecord], check_shapes: bool = True) -> None:
    if not check_shapes:
        return

    shape_mismatches = []
    for record in tqdm(records, desc="validating headers", unit="scan"):
        image_shape = nib.load(record.image_path).shape
        mask_shape = nib.load(record.mask_path).shape
        if image_shape != mask_shape:
            shape_mismatches.append((record.key, image_shape, mask_shape))

    if shape_mismatches:
        examples = "\n".join(
            f"  {key}: image={image_shape}, mask={mask_shape}"
            for key, image_shape, mask_shape in shape_mismatches[:10]
        )
        raise ValueError(f"{len(shape_mismatches)} image/mask shape mismatches:\n{examples}")


def report_records(records: list[AABCRecord]) -> None:
    matched = sum(record.metadata is not None for record in records)
    modalities = sorted({record.modality for record in records})
    print(f"records: {len(records)}")
    print(f"metadata matched: {matched}/{len(records)}")
    print(f"modalities: {', '.join(modalities)}")


def record_to_sample(record: AABCRecord) -> dict[str, Any]:
    image_img = nib.load(record.image_path)
    mask_img = nib.load(record.mask_path)

    image = np.asarray(image_img.get_fdata(dtype=np.float32), dtype=np.float32)
    mask = np.asarray(mask_img.dataobj) > 0
    mask = mask.astype(np.uint8, copy=False)

    if image.shape != mask.shape:
        raise ValueError(f"{record.key}: image shape {image.shape} != mask shape {mask.shape}")

    meta = make_meta(record, image_img=image_img, mask_img=mask_img, image=image, mask=mask)
    return {
        "__key__": record.key,
        "image.npy": image,
        "mask.npy": mask,
        "meta.json": meta,
    }


def make_meta(
    record: AABCRecord,
    image_img: nib.Nifti1Image,
    mask_img: nib.Nifti1Image,
    image: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    meta = {
        "scan_id": record.key,
        "image_path": str(record.image_path),
        "mask_path": str(record.mask_path),
        "subject": record.subject,
        "subject_id": record.subject_id,
        "session": record.session,
        "visit": record.visit,
        "modality": record.modality,
        "id_event": record.id_event,
        "shape": list(image.shape),
        "mask_shape": list(mask.shape),
        "image_dtype": str(image.dtype),
        "mask_dtype": str(mask.dtype),
        "affine": image_img.affine.tolist(),
        "header": header_summary(image_img),
        "mask_header": header_summary(mask_img),
        "metadata": record.metadata,
    }

    row = record.metadata or {}
    for field in CORE_METADATA_FIELDS:
        meta[field] = row.get(field)
    return meta


def header_summary(img: nib.Nifti1Image) -> dict[str, Any]:
    header = img.header
    qform_code = int(header["qform_code"])
    sform_code = int(header["sform_code"])
    xyzt_units = header.get_xyzt_units()
    descrip = header["descrip"].item()
    if isinstance(descrip, bytes):
        descrip = descrip.decode("utf-8", errors="replace").rstrip("\x00")

    return {
        "shape": list(img.shape),
        "zooms": [float(value) for value in header.get_zooms()[: len(img.shape)]],
        "qform_code": qform_code,
        "sform_code": sform_code,
        "xyzt_units": list(xyzt_units),
        "datatype": str(header.get_data_dtype()),
        "descrip": descrip,
    }


def scan_key(path: Path) -> str:
    name = path.name
    if not name.endswith(PROCESSED_SUFFIX):
        raise ValueError(f"unexpected processed filename: {name}")
    return name[: -len(PROCESSED_SUFFIX)]


def coerce_csv_value(value: str | None) -> Any:
    if value is None:
        return None
    value = value.strip()
    if value == "":
        return None
    try:
        parsed = float(value)
    except ValueError:
        return value
    if not math.isfinite(parsed):
        return None
    if parsed.is_integer() and re.fullmatch(r"[+-]?\d+(\.0+)?", value):
        return int(parsed)
    return parsed


def assert_output_ready(output_pattern: str, overwrite: bool = False) -> None:
    existing = sorted(glob(output_pattern.replace("%06d", "*").replace("%d", "*")))
    if existing and not overwrite:
        examples = "\n".join(f"  {path}" for path in existing[:10])
        raise FileExistsError(
            f"{len(existing)} output shard(s) already match {output_pattern!r}; "
            f"use --overwrite to replace them.\n{examples}"
        )


if __name__ == "__main__":
    main()
