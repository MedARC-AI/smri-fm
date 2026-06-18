from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


MANIFEST_FIELDS = [
    "image_path",
    "condition",
    "modality",
    "modality_code",
    "plane",
    "dim",
    "spacing",
    "fov",
    "target_output_subdir",
    "generation_seed",
    "qc_path",
    "qc_min",
    "qc_mean",
    "qc_metric",
    "qc_threshold",
    "qc_pass",
    "qc_error",
]


def write_manifest(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(_stringify_record(record))


def write_accepted_manifest(path: Path, records: list[dict[str, Any]]) -> None:
    accepted = [record for record in records if record.get("qc_pass") is True]
    write_manifest(path, accepted)


def _stringify_record(record: dict[str, Any]) -> dict[str, str]:
    row: dict[str, str] = {}
    for field in MANIFEST_FIELDS:
        value = record.get(field)
        if value is None:
            row[field] = ""
        elif isinstance(value, bool):
            row[field] = "true" if value else "false"
        else:
            row[field] = str(value)
    return row

