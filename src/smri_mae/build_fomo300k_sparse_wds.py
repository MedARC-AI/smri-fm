#!/usr/bin/env python3
# Create leakage-safe, QC-filtered, z-score-normalized FP16 FOMO300 shards.

from __future__ import annotations

import argparse
import csv
import io
import os
import json
import math
import random
import re
import shutil
import tarfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm


INTENSITY_NORM_EPS = 1e-6

ENTITY_RE = re.compile(r"^(sub|ses|run|acq|dir|echo|rec|part|task|space|desc)-", re.IGNORECASE)


@dataclass(frozen=True)
class Candidate:
    key: str
    subset: str
    image_path: Path
    mask_path: Path
    modality: str
    native_stem: str
    dwi_family: str | None = None


def is_dwi_family(native_stem: str) -> bool:
    return "dwi" in native_stem.lower()


def family_from_stem(stem: str) -> str:
    tokens = stem.removesuffix("_qc").split("_")
    content = [token for token in tokens if not ENTITY_RE.match(token)]
    lowered = [token.lower() for token in content]

    patterns = (
        ("b0", ("bval0", "bval-0", "b0")),
        ("b100", ("bval100", "bval-100", "b100")),
        ("b500", ("bval500", "bval-500", "b500")),
        ("b800", ("bval800", "bval-800", "b800")),
        ("b1000", ("bval1000", "bval-1000", "b1000")),
        ("b1500", ("bval1500", "bval-1500", "b1500")),
        ("b2000", ("bval2000", "bval-2000", "b2000")),
        ("b2500", ("bval2500", "bval-2500", "b2500")),
        ("b3000", ("bval3000", "bval-3000", "b3000")),
        ("ADC", ("adc",)),
        ("FA", ("fa",)),
        ("MD", ("md",)),
        ("RD", ("rd",)),
        ("AD", ("ad",)),
        ("TRACE", ("trace",)),
        ("DTI", ("dti",)),
    )
    for label, aliases in patterns:
        if any(token in aliases for token in lowered):
            return label

    dwi_tokens = [token for token in content if "dwi" in token.lower()]
    if dwi_tokens:
        return dwi_tokens[-1]
    return "DWI-other"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create sparse brain-voxel WebDataset shards from processed NIfTIs."
    )
    parser.add_argument("--src-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--maxsize", type=int, default=3_000_000_000)
    parser.add_argument("--maxcount", type=int, default=150)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of parallel shard-writing workers. Values >1 write independent "
            "shard.%%06d.tar files concurrently; each shard contains at most --maxcount samples."
        ),
    )
    parser.add_argument("--template-space", default="MNI152NLin2009cAsym")
    parser.add_argument("--image-shape", nargs=3, type=int, default=[208, 240, 208])
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=["all"],
        help="Modalities to include, or 'all' to include every processed file.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle discovered candidates before sharding so shards mix cohorts/modalities.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for --shuffle; ignored without --shuffle.",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help=(
            "Fraction of *subjects* (not scans) held out for val. Split is subject-level so "
            "all sessions of a subject go to the same split (no session leakage). "
            "Set 0 to write a single unsplit shard tree."
        ),
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="RNG seed for subject-level train/val assignment.",
    )
    parser.add_argument("--skip-manifest", type=Path, default=None)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument(
        "--qc-report",
        type=Path,
        default=None,
        help="Optional QC TSV/CSV with key, acquisition_label, and final_decision columns to embed in meta.json.",
    )
    parser.add_argument(
        "--min-synthseg-qc",
        type=float,
        default=None,
        help="Only include samples with mean SynthSeg regional QC strictly greater than this value.",
    )
    return parser.parse_args()


SUB_RE = re.compile(r"(sub-[A-Za-z0-9]+)")
DS_RE = re.compile(r"(ds\d+)", re.IGNORECASE)


def subject_unit_id(candidate: Candidate) -> str:
    sub_m = SUB_RE.search(candidate.native_stem) or SUB_RE.search(candidate.key)
    if sub_m is None:
        raise ValueError(f"could not parse subject from candidate key={candidate.key!r}")
    subject = sub_m.group(1)
    ds_m = DS_RE.search(candidate.key) or DS_RE.search(candidate.native_stem)
    if candidate.subset.startswith("PT030_OpenNeuro") and ds_m is not None:
        return f"{candidate.subset}/{ds_m.group(1).lower()}/{subject}"
    return f"{candidate.subset}/{subject}"


def assign_subject_splits(
    candidates: list[Candidate],
    *,
    val_ratio: float,
    split_seed: int,
) -> tuple[list[Candidate], list[Candidate], dict[str, str]]:
    if not (0.0 <= val_ratio < 1.0):
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")

    unit_ids = sorted({subject_unit_id(c) for c in candidates})
    rng = random.Random(split_seed)
    rng.shuffle(unit_ids)

    if val_ratio <= 0.0 or len(unit_ids) < 2:
        unit_to_split = {u: "train" for u in unit_ids}
    else:
        n_val = int(round(len(unit_ids) * val_ratio))
        n_val = max(1, min(n_val, len(unit_ids) - 1))
        val_units = set(unit_ids[:n_val])
        unit_to_split = {u: ("val" if u in val_units else "train") for u in unit_ids}

    train: list[Candidate] = []
    val: list[Candidate] = []
    for c in candidates:
        if unit_to_split[subject_unit_id(c)] == "val":
            val.append(c)
        else:
            train.append(c)
    return train, val, unit_to_split


def processed_native_stem(path: Path, template_space: str) -> str:
    suffix = f"_space-{template_space}_desc-processed.nii.gz"
    if not path.name.endswith(suffix):
        raise ValueError(f"unexpected processed filename: {path.name}")
    return path.name[: -len(suffix)]


def key_for(subset: str, native_stem: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{subset}_{native_stem}")


def read_synthseg_mean_qc(path: Path) -> float:
    values: list[float] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, value in row.items():
                if key is None or key == "Unnamed: 0" or value in {None, ""}:
                    continue
                try:
                    val = float(value)
                except ValueError:
                    continue
                if math.isfinite(val):
                    values.append(val)
    if not values:
        raise ValueError("no numeric SynthSeg QC values")
    return float(sum(values) / len(values))


def skip_record(
    image_path: Path,
    reason: str,
    detail: str,
    *,
    subset: str | None = None,
    modality: str | None = None,
) -> dict[str, Any]:
    return {
        "image_path": str(image_path),
        "subset": subset,
        "modality": modality,
        "reason": reason,
        "detail": detail,
    }


def _list_pt_inventory(payload: tuple[str, str, bool]) -> dict[str, Any]:
    pt_dir_s, template_space, need_synthseg = payload
    pt = Path(pt_dir_s)
    proc = pt / "processed"
    mask_dir = pt / "derivatives" / "masks"
    synth_dir = pt / "derivatives" / "synthseg"
    processed: list[str] = []
    if proc.is_dir():
        pattern = f"*_space-{template_space}_desc-processed.nii.gz"
        try:
            processed = [str(p) for p in sorted(proc.glob(pattern))]
        except OSError:
            processed = []
    masks: set[str] = set()
    if mask_dir.is_dir():
        try:
            masks = {p.name for p in mask_dir.iterdir() if p.is_file()}
        except OSError:
            masks = set()
    qc: set[str] = set()
    if need_synthseg and synth_dir.is_dir():
        try:
            for p in synth_dir.iterdir():
                if not p.is_file():
                    continue
                name = p.name
                if name.endswith("_qc.csv"):
                    qc.add(name)
        except OSError:
            pass
    return {
        "pt": pt.name,
        "processed": processed,
        "masks": masks,
        "qc": qc,
    }


def discover_candidates(
    src_root: Path,
    *,
    template_space: str,
    allowed_modalities: set[str],
    max_samples: int | None,
    min_synthseg_qc: float | None = None,
    workers: int = 1,
) -> tuple[list[Candidate], list[dict[str, Any]]]:
    candidates: list[Candidate] = []
    skipped: list[dict[str, Any]] = []
    seen_keys: Counter[str] = Counter()
    allowed = {m.lower() for m in allowed_modalities}
    workers = max(1, int(workers))
    list_workers = min(workers, 32)

    pt_dirs = sorted(p for p in src_root.iterdir() if p.is_dir() and p.name.startswith("PT"))
    need_synthseg = min_synthseg_qc is not None
    list_jobs = [(str(p), template_space, need_synthseg) for p in pt_dirs]
    print(
        f"inventory under {len(pt_dirs)} PT folders "
        f"(list_workers={list_workers}, write_workers={workers}, "
        f"need_synthseg={need_synthseg})..."
    )

    inventories: list[dict[str, Any]] = []
    if list_workers == 1:
        for job in tqdm(list_jobs, desc="list PT inventory"):
            inventories.append(_list_pt_inventory(job))
    else:
        with ProcessPoolExecutor(max_workers=list_workers) as pool:
            for inv in tqdm(
                pool.map(_list_pt_inventory, list_jobs, chunksize=1),
                total=len(list_jobs),
                desc="list PT inventory",
            ):
                inventories.append(inv)

    n_paths = sum(len(inv["processed"]) for inv in inventories)
    print(f"found {n_paths} processed paths; matching masks in memory...")

    for inv in tqdm(inventories, desc="build candidates"):
        masks: set[str] = inv["masks"]
        qc_names: set[str] = inv["qc"]
        for image_path_s in inv["processed"]:
            image_path = Path(image_path_s)
            try:
                native_stem = processed_native_stem(image_path, template_space)
            except ValueError as exc:
                skipped.append(skip_record(image_path, "unexpected_processed_name", str(exc)))
                continue

            if is_dwi_family(native_stem):
                modality = "dwi"
                dwi_family = family_from_stem(native_stem)
            else:
                modality = native_stem.rsplit("_", 1)[-1].lower()
                dwi_family = None
            if "all" not in allowed and modality not in allowed:
                continue

            if min_synthseg_qc is not None:
                qc_name = f"{native_stem}_qc.csv"
                if qc_name not in qc_names:
                    skipped.append(
                        skip_record(
                            image_path,
                            "missing_synthseg_qc",
                            f"missing SynthSeg QC CSV: {qc_name}",
                            subset=inv["pt"],
                            modality=modality,
                        )
                    )
                    continue
                qc_path = image_path.parent.parent / "derivatives" / "synthseg" / qc_name
                try:
                    mean_qc = read_synthseg_mean_qc(qc_path)
                except Exception as exc:
                    skipped.append(
                        skip_record(
                            image_path,
                            "invalid_synthseg_qc",
                            str(exc),
                            subset=inv["pt"],
                            modality=modality,
                        )
                    )
                    continue
                if mean_qc <= min_synthseg_qc:
                    skipped.append(
                        skip_record(
                            image_path,
                            "low_synthseg_qc",
                            f"mean_qc={mean_qc:.6f} <= {min_synthseg_qc:.6f}",
                            subset=inv["pt"],
                            modality=modality,
                        )
                    )
                    continue

            mask_name = f"{native_stem}_space-{template_space}_desc-brain_mask.nii.gz"
            subset = inv["pt"]
            if mask_name in masks:
                mask_path = image_path.parent.parent / "derivatives" / "masks" / mask_name
            else:
                skipped.append(
                    skip_record(
                        image_path,
                        "missing_mask_source",
                        "missing derivatives/masks brain mask",
                        subset=subset,
                        modality=modality,
                    )
                )
                continue

            candidate = Candidate(
                key=key_for(subset, native_stem),
                subset=subset,
                image_path=image_path,
                mask_path=mask_path,
                modality=modality,
                native_stem=native_stem,
                dwi_family=dwi_family,
            )
            seen_keys[candidate.key] += 1
            if seen_keys[candidate.key] > 1:
                skipped.append(
                    skip_record(
                        image_path,
                        "duplicate_key",
                        f"duplicate WebDataset key {candidate.key!r}",
                        subset=subset,
                        modality=modality,
                    )
                )
                continue
            candidates.append(candidate)
            if max_samples is not None and len(candidates) >= max_samples:
                return candidates, skipped

    return candidates, skipped


def read_qc_report(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(path)
    import pandas as pd

    sep = "\t" if path.suffix == ".tsv" else ","
    df = pd.read_csv(path, sep=sep, low_memory=False)
    if "key" not in df:
        raise ValueError(f"QC report has no key column: {path}")
    keep_cols = [
        "quality_decision",
        "quality_reasons",
        "quality_review_score",
        "artifact_tags",
        "final_decision",
        "final_reason",
        "manual_qc_label",
        "combined_acquisition_label",
        "combined_acquisition_source",
        "native_header_acquisition_label",
        "native_header_acquisition_reason",
        "native_voxel_size_x",
        "native_voxel_size_y",
        "native_voxel_size_z",
        "native_spacing_ratio",
        "native_min_dim",
        "acquisition_label",
        "acquisition_is_likely_2d",
        "acquisition_label_reason",
        "p_bad_tabular",
        "p_bad_image",
        "review_score",
        "soft_failure_count",
        "soft_failure_reasons",
    ]
    out: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        item: dict[str, Any] = {}
        for col in keep_cols:
            if col not in df:
                continue
            value = row[col]
            if pd.isna(value):
                continue
            if isinstance(value, np.generic):
                value = value.item()
            item[col] = value
        out[str(row["key"])] = item
    return out


def load_volume(path: Path, *, dtype: np.dtype) -> np.ndarray:
    import nibabel as nib

    img = nib.load(path)
    data = np.asanyarray(img.dataobj)
    return np.asarray(data, dtype=dtype)


def fit_to_volume_shape(
    data: np.ndarray,
    volume_shape: tuple[int, int, int],
    *,
    pad_value: int | float | bool = 0,
) -> np.ndarray:
    if data.ndim != 3:
        raise ValueError(f"expected 3D volume, got shape {data.shape}")
    if data.shape == volume_shape:
        return data

    src_slices = []
    dst_slices = []
    for src_dim, dst_dim in zip(data.shape, volume_shape, strict=True):
        if src_dim >= dst_dim:
            src_start = (src_dim - dst_dim) // 2
            src_end = src_start + dst_dim
            dst_start = 0
            dst_end = dst_dim
        else:
            src_start = 0
            src_end = src_dim
            dst_start = (dst_dim - src_dim) // 2
            dst_end = dst_start + src_dim
        src_slices.append(slice(src_start, src_end))
        dst_slices.append(slice(dst_start, dst_end))

    fitted = np.full(volume_shape, pad_value, dtype=data.dtype)
    fitted[tuple(dst_slices)] = data[tuple(src_slices)]
    return fitted


def zscore_brain_values(values: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    values_f32 = np.asarray(values, dtype=np.float32)
    if values_f32.size == 0:
        raise ValueError("cannot normalize empty brain value array")
    if not bool(np.isfinite(values_f32).all()):
        raise ValueError("brain values contain NaN or inf")

    raw_mean = float(values_f32.mean(dtype=np.float64))
    raw_std = float(values_f32.std(dtype=np.float64))
    normalization_scale = raw_std if raw_std >= INTENSITY_NORM_EPS else 1.0
    normalized = (values_f32 - raw_mean) / normalization_scale
    metadata = {
        "method": "brain_mask_zscore",
        "scope": "masked_brain_voxels_after_shape_fit",
        "raw_mean": raw_mean,
        "raw_std": raw_std,
        "normalization_scale": float(normalization_scale),
        "eps": INTENSITY_NORM_EPS,
        "inverse": "raw = normalized * normalization_scale + raw_mean",
    }
    return np.ascontiguousarray(normalized, dtype=np.float16), metadata


def make_sample(
    candidate: Candidate,
    volume_shape: tuple[int, int, int],
    qc_metadata: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], int]:
    image = load_volume(candidate.image_path, dtype=np.float32)
    mask = load_volume(candidate.mask_path, dtype=np.uint8) > 0
    source_shape = tuple(int(dim) for dim in image.shape)
    if image.shape != mask.shape:
        raise ValueError(f"image shape {image.shape} != mask shape {mask.shape}")
    image = fit_to_volume_shape(image, volume_shape, pad_value=0)
    mask = fit_to_volume_shape(mask, volume_shape, pad_value=False)
    if not bool(mask.any()):
        raise ValueError("empty brain mask")

    image = image[None, ...]
    mask = mask[None, ...]
    image_values, intensity_normalization = zscore_brain_values(image[mask])
    img_mask = np.packbits(mask.reshape(-1).astype(np.uint8))
    voxel_count = int(mask.sum())
    if image_values.size != voxel_count:
        raise ValueError("sparse value count does not match mask voxel count")

    qc_metadata = dict(qc_metadata or {})
    meta = {
        "key": candidate.key,
        "subset": candidate.subset,
        "modality": candidate.modality,
        "dwi_family": candidate.dwi_family,
        "native_stem": candidate.native_stem,
        "image_path": str(candidate.image_path),
        "mask_source_path": str(candidate.mask_path),
        "mask_source_type": "brain_mask",
        "acquisition_label": qc_metadata.get("acquisition_label", "unknown"),
        "quality_decision": qc_metadata.get("quality_decision", "unknown"),
        "raw_mean": intensity_normalization["raw_mean"],
        "raw_std": intensity_normalization["raw_std"],
        "normalization_scale": intensity_normalization["normalization_scale"],
        "intensity_normalization": intensity_normalization,
        "qc": qc_metadata,
        "sparse_image": {
            "scheme": "mask_selected_values",
            "source": "img_mask",
            "source_shape": list(source_shape),
            "dense_shape": [1, *volume_shape],
            "shape_fit": "center_crop_or_pad",
            "values_dtype": "float16",
            "values_normalized": True,
            "num_voxels": voxel_count,
        },
    }
    return (
        {
            "__key__": candidate.key,
            "image_values.npy": image_values,
            "img_mask.npy": np.ascontiguousarray(img_mask, dtype=np.uint8),
            "meta.json": meta,
        },
        voxel_count,
    )


def npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def json_bytes(obj: dict[str, Any]) -> bytes:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True).encode("utf-8")


def add_tar_file(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    info.mode = 0o644
    tar.addfile(info, io.BytesIO(data))


def write_sample_to_tar(tar: tarfile.TarFile, sample: dict[str, Any]) -> None:
    key = sample["__key__"]
    add_tar_file(tar, f"{key}.image_values.npy", npy_bytes(sample["image_values.npy"]))
    add_tar_file(tar, f"{key}.img_mask.npy", npy_bytes(sample["img_mask.npy"]))
    add_tar_file(tar, f"{key}.meta.json", json_bytes(sample["meta.json"]))


def prepare_output(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"output already exists: {path} (pass --overwrite to replace)")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def build_metadata(
    *,
    args: argparse.Namespace,
    candidates: list[Candidate],
    skipped: list[dict[str, Any]],
    written: int,
    voxel_counts: list[int],
    qc_by_key: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    voxel_arr = np.asarray(voxel_counts, dtype=np.int64)
    metadata: dict[str, Any] = {
        "src_root": str(args.src_root),
        "out": str(args.out),
        "format": "sparse_wds",
        "image_values_dtype": "float16",
        "image_values_normalized": True,
        "intensity_normalization": {
            "method": "brain_mask_zscore",
            "scope": "per_sample_masked_brain_voxels_after_shape_fit",
            "eps": INTENSITY_NORM_EPS,
            "inverse": "raw = normalized * normalization_scale + raw_mean",
        },
        "template_space": args.template_space,
        "image_shape": [1, *args.volume_shape],
        "mask_packed": True,
        "maxsize": args.maxsize,
        "maxcount": args.maxcount,
        "workers": args.workers,
        "requested_modalities": sorted(args.allowed_modalities),
        "shuffle": bool(args.shuffle),
        "seed": args.seed if args.shuffle else None,
        "discovered_samples": len(candidates),
        "written_samples": written,
        "skipped_samples": len(skipped),
        "modality_counts": dict(sorted(Counter(c.modality for c in candidates).items())),
        "dwi_family_counts": dict(
            sorted(Counter(c.dwi_family for c in candidates if c.dwi_family is not None).items())
        ),
        "mask_source_counts": {"brain_mask": len(candidates)},
        "skip_reason_counts": dict(sorted(Counter(s["reason"] for s in skipped).items())),
        "qc_report": str(args.qc_report) if args.qc_report else None,
        "min_synthseg_qc": args.min_synthseg_qc,
    }
    if qc_by_key is not None:
        acquisition_counts = Counter(
            qc_by_key.get(c.key, {}).get("acquisition_label", "unknown") for c in candidates
        )
        metadata["acquisition_label_counts"] = dict(sorted(acquisition_counts.items()))
    if len(voxel_arr) > 0:
        metadata.update(
            {
                "sparse_num_voxels_min": int(voxel_arr.min()),
                "sparse_num_voxels_mean": float(voxel_arr.mean()),
                "sparse_num_voxels_max": int(voxel_arr.max()),
            }
        )
    return metadata


def build_wds(
    args: argparse.Namespace,
    candidates: list[Candidate],
    skipped: list[dict[str, Any]],
    qc_by_key: dict[str, dict[str, Any]],
    *,
    out_dir: Path | None = None,
    desc: str = "sparse wds",
) -> tuple[int, list[int], list[dict[str, Any]], set[str]]:
    out_dir = Path(out_dir or args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    shard_idx = 0
    shard_count = 0
    voxel_counts: list[int] = []
    written_keys: set[str] = set()
    tar: tarfile.TarFile | None = None
    try:
        for candidate in tqdm(candidates, desc=desc, total=len(candidates)):
            if tar is None or shard_count >= args.maxcount:
                if tar is not None:
                    tar.close()
                tar = tarfile.open(out_dir / f"shard.{shard_idx:06d}.tar", "w")
                shard_idx += 1
                shard_count = 0
            try:
                sample, voxel_count = make_sample(
                    candidate,
                    args.volume_shape,
                    qc_by_key.get(candidate.key),
                )
            except Exception as exc:
                skipped.append(
                    skip_record(
                        candidate.image_path,
                        "sample_error",
                        str(exc),
                        subset=candidate.subset,
                        modality=candidate.modality,
                    )
                )
                continue
            write_sample_to_tar(tar, sample)
            voxel_counts.append(voxel_count)
            written_keys.add(candidate.key)
            written += 1
            shard_count += 1
    finally:
        if tar is not None:
            tar.close()
    return written, voxel_counts, skipped, written_keys


def candidate_shard_jobs(
    candidates: list[Candidate],
    qc_by_key: dict[str, dict[str, Any]],
    *,
    maxcount: int,
) -> list[tuple[int, list[tuple[Candidate, dict[str, Any] | None]]]]:
    if maxcount <= 0:
        raise ValueError(f"maxcount must be positive, got {maxcount}")
    jobs = []
    for shard_idx, start in enumerate(range(0, len(candidates), maxcount)):
        chunk = candidates[start : start + maxcount]
        jobs.append((shard_idx, [(candidate, qc_by_key.get(candidate.key)) for candidate in chunk]))
    return jobs


def write_wds_shard_job(
    shard_idx: int,
    items: list[tuple[Candidate, dict[str, Any] | None]],
    out: Path,
    maxsize: int,
    volume_shape: tuple[int, int, int],
) -> dict[str, Any]:
    final_path = out / f"shard.{shard_idx:06d}.tar"
    tmp_path = out / f".shard.{shard_idx:06d}.{os.getpid()}.tar.tmp"
    written = 0
    voxel_counts: list[int] = []
    skipped: list[dict[str, Any]] = []
    written_keys: list[str] = []
    try:
        with tarfile.open(tmp_path, "w") as writer:
            for candidate, qc_metadata in items:
                try:
                    sample, voxel_count = make_sample(candidate, volume_shape, qc_metadata)
                except Exception as exc:
                    skipped.append(
                        skip_record(
                            candidate.image_path,
                            "sample_error",
                            str(exc),
                            subset=candidate.subset,
                            modality=candidate.modality,
                        )
                    )
                    continue
                write_sample_to_tar(writer, sample)
                voxel_counts.append(voxel_count)
                written_keys.append(candidate.key)
                written += 1
        if written == 0:
            tmp_path.unlink(missing_ok=True)
            size_bytes = 0
        else:
            size_bytes = tmp_path.stat().st_size
            if size_bytes > maxsize:
                raise RuntimeError(
                    f"parallel shard {final_path.name} is {size_bytes} bytes, above maxsize {maxsize}"
                )
            tmp_path.replace(final_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return {
        "shard_idx": shard_idx,
        "written": written,
        "written_keys": written_keys,
        "voxel_counts": voxel_counts,
        "skipped": skipped,
        "size_bytes": size_bytes,
    }


def build_wds_parallel(
    args: argparse.Namespace,
    candidates: list[Candidate],
    skipped: list[dict[str, Any]],
    qc_by_key: dict[str, dict[str, Any]],
    *,
    out_dir: Path | None = None,
    desc: str = "sparse wds shards",
) -> tuple[int, list[int], list[dict[str, Any]], set[str]]:
    out_dir = Path(out_dir or args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = candidate_shard_jobs(candidates, qc_by_key, maxcount=args.maxcount)
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                write_wds_shard_job,
                shard_idx,
                items,
                out_dir,
                args.maxsize,
                args.volume_shape,
            )
            for shard_idx, items in jobs
        ]
        for future in tqdm(as_completed(futures), desc=desc, total=len(futures)):
            results.append(future.result())

    written = 0
    voxel_counts: list[int] = []
    written_keys: set[str] = set()
    for result in sorted(results, key=lambda item: item["shard_idx"]):
        written += int(result["written"])
        voxel_counts.extend(result["voxel_counts"])
        skipped.extend(result["skipped"])
        written_keys.update(result["written_keys"])
    return written, voxel_counts, skipped, written_keys


def _write_split_candidates(
    path: Path,
    candidates: list[Candidate],
    *,
    split: str,
    unit_to_split: dict[str, str],
) -> None:
    rows = []
    for c in candidates:
        unit = subject_unit_id(c)
        rows.append(
            {
                "key": c.key,
                "subset": c.subset,
                "modality": c.modality,
                "dwi_family": c.dwi_family,
                "image_path": str(c.image_path),
                "split": split,
                "subject_unit": unit,
            }
        )
    write_jsonl(path, rows)


def main() -> None:
    args = parse_args()
    args.src_root = args.src_root.resolve()
    args.out = args.out.resolve()
    args.skip_manifest = (args.skip_manifest or (args.out / "skipped.jsonl")).resolve()
    args.metadata = (args.metadata or (args.out / "metadata.json")).resolve()
    args.volume_shape = tuple(args.image_shape)
    args.allowed_modalities = {m.lower() for m in args.modalities}

    if not args.src_root.exists():
        raise FileNotFoundError(args.src_root)
    candidates, skipped = discover_candidates(
        args.src_root,
        template_space=args.template_space,
        allowed_modalities=args.allowed_modalities,
        max_samples=args.max_samples,
        min_synthseg_qc=args.min_synthseg_qc,
        workers=args.workers,
    )
    qc_by_key = read_qc_report(args.qc_report)

    train_cands, val_cands, unit_to_split = assign_subject_splits(
        candidates,
        val_ratio=float(args.val_ratio),
        split_seed=int(args.split_seed),
    )
    n_units = len(unit_to_split)
    n_val_units = sum(1 for s in unit_to_split.values() if s == "val")
    print(
        f"subject split: units={n_units} train_units={n_units - n_val_units} "
        f"val_units={n_val_units} (val_ratio={args.val_ratio}, seed={args.split_seed})"
    )
    print(f"scan split: train={len(train_cands)} val={len(val_cands)} total={len(candidates)}")

    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(train_cands)
        rng.shuffle(val_cands)

    if args.dry_run:
        metadata = build_metadata(
            args=args,
            candidates=candidates,
            skipped=skipped,
            written=0,
            voxel_counts=[],
            qc_by_key=qc_by_key,
        )
        metadata["split"] = {
            "val_ratio": args.val_ratio,
            "split_seed": args.split_seed,
            "n_subject_units": n_units,
            "n_train_units": n_units - n_val_units,
            "n_val_units": n_val_units,
            "n_train_scans": len(train_cands),
            "n_val_scans": len(val_cands),
        }
        print(json.dumps(metadata, indent=2, sort_keys=True))
        return

    if not candidates:
        raise RuntimeError("no eligible samples found")

    if args.workers < 1:
        raise ValueError(f"workers must be at least 1, got {args.workers}")

    prepare_output(args.out, overwrite=args.overwrite)

    written = 0
    voxel_counts: list[int] = []
    split_stats: dict[str, Any] = {}
    written_by_split: dict[str, list[Candidate]] = {}

    def _run_split(name: str, cands: list[Candidate]) -> None:
        nonlocal written, voxel_counts, skipped, written_by_split
        if not cands:
            split_stats[name] = {"written": 0, "n_candidates": 0, "n_manifest_rows": 0}
            written_by_split[name] = []
            return

        out_dir = args.out / name if args.val_ratio > 0 else args.out
        if args.workers == 1:
            w, vc, skipped, written_keys = build_wds(
                args, cands, skipped, qc_by_key, out_dir=out_dir, desc=f"sparse wds [{name}]"
            )
        else:
            w, vc, skipped, written_keys = build_wds_parallel(
                args,
                cands,
                skipped,
                qc_by_key,
                out_dir=out_dir,
                desc=f"sparse wds shards [{name}]",
            )
        written += w
        voxel_counts.extend(vc)
        written_candidates = [c for c in cands if c.key in written_keys]
        written_by_split[name] = written_candidates
        split_stats[name] = {
            "written": w,
            "n_candidates": len(cands),
            "n_manifest_rows": len(written_candidates),
            "out_dir": str(out_dir),
        }
        _write_split_candidates(
            args.out / f"candidates_{name}.jsonl",
            written_candidates,
            split=name,
            unit_to_split=unit_to_split,
        )

    if args.val_ratio > 0:
        _run_split("train", train_cands)
        _run_split("val", val_cands)
    else:
        _run_split("train", train_cands)

    all_rows = []
    for split_name, cands in (
        ("train", written_by_split.get("train", [])),
        ("val", written_by_split.get("val", [])),
    ):
        for c in cands:
            all_rows.append(
                {
                    "key": c.key,
                    "subset": c.subset,
                    "modality": c.modality,
                    "dwi_family": c.dwi_family,
                    "image_path": str(c.image_path),
                    "split": split_name if args.val_ratio > 0 else "train",
                    "subject_unit": subject_unit_id(c),
                }
            )
    write_jsonl(args.out / "candidates.jsonl", all_rows)

    write_jsonl(
        args.out / "subject_splits.jsonl",
        [{"subject_unit": unit, "split": split} for unit, split in sorted(unit_to_split.items())],
    )

    write_jsonl(args.skip_manifest, skipped)
    metadata = build_metadata(
        args=args,
        candidates=candidates,
        skipped=skipped,
        written=written,
        voxel_counts=voxel_counts,
        qc_by_key=qc_by_key,
    )
    metadata["split"] = {
        "val_ratio": args.val_ratio,
        "split_seed": args.split_seed,
        "n_subject_units": n_units,
        "n_train_units": n_units - n_val_units,
        "n_val_units": n_val_units,
        "n_train_scans": len(written_by_split.get("train", [])),
        "n_val_scans": len(written_by_split.get("val", [])),
        "leakage_policy": "subject_unit_all_sessions_same_split",
        "subject_unit_definition": (
            "OpenNeuro: PT030_OpenNeuro/<ds>/<sub-*>; else <subset>/<sub-*>"
        ),
        "splits": split_stats,
    }
    write_metadata(args.metadata, metadata)
    print(
        f"wrote {written} samples to {args.out}; "
        f"skipped {len(skipped)} samples; metadata={args.metadata}"
    )
    print(f"split stats: {json.dumps(split_stats)}")


if __name__ == "__main__":
    main()
