from __future__ import annotations

import csv
import logging
import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .config import PipelineConfig

log = logging.getLogger("synthetic_pipeline.qc")


def run_qc(cfg: PipelineConfig, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not records:
        return records
    if cfg.qc.mode == "direct_synthseg":
        run_direct_synthseg(cfg, records)
    elif cfg.qc.mode == "preprocess_then_synthseg":
        run_preprocessing_pipeline(cfg)
        attach_preprocessing_qc_paths(cfg, records)
    else:
        raise ValueError(f"Unsupported QC mode: {cfg.qc.mode}")

    for record in records:
        apply_qc_threshold(cfg, record)
    return records


def run_direct_synthseg(cfg: PipelineConfig, records: list[dict[str, Any]]) -> None:
    tasks = []
    for record in records:
        image_path = Path(record["image_path"])
        target_dir = cfg.output_dir / "derivatives" / "synthseg" / record["target_output_subdir"]
        target_dir.mkdir(parents=True, exist_ok=True)
        stem = image_path.name.removesuffix(".nii.gz")
        seg_path = target_dir / f"{stem}_desc-synthseg_dseg.nii.gz"
        vol_path = target_dir / f"{stem}_volumes.csv"
        qc_path = target_dir / f"{stem}_qc.csv"
        record["qc_path"] = str(qc_path.resolve())
        tasks.append((image_path, seg_path, vol_path, qc_path))

    pending = [
        task for task in tasks
        if not (task[1].exists() and task[2].exists() and task[3].exists())
    ]
    if not pending:
        log.info("SynthSeg QC outputs already exist for all generated images.")
        return

    input_paths = [task[0] for task in pending]
    seg_paths = [task[1] for task in pending]
    vol_paths = [task[2] for task in pending]
    qc_paths = [task[3] for task in pending]
    run_synthseg(
        input_paths,
        seg_paths,
        vol_paths,
        qc_paths,
        cfg.qc.synthseg_cmd,
        cfg.qc.threads,
        cfg.qc.cpu,
    )


def run_synthseg(
    input_paths: list[Path],
    seg_paths: list[Path],
    vol_paths: list[Path],
    qc_paths: list[Path],
    synthseg_cmd: str,
    threads: int,
    cpu_only: bool,
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        input_txt = tmp_path / "inputs.txt"
        output_txt = tmp_path / "outputs.txt"
        vol_txt = tmp_path / "volumes.txt"
        qc_txt = tmp_path / "qc.txt"
        input_txt.write_text("\n".join(str(p) for p in input_paths) + "\n")
        output_txt.write_text("\n".join(str(p) for p in seg_paths) + "\n")
        vol_txt.write_text("\n".join(str(p) for p in vol_paths) + "\n")
        qc_txt.write_text("\n".join(str(p) for p in qc_paths) + "\n")

        cmd = shlex.split(synthseg_cmd) + [
            "--i",
            str(input_txt),
            "--o",
            str(output_txt),
            "--vol",
            str(vol_txt),
            "--qc",
            str(qc_txt),
            "--parc",
            "--robust",
            "--threads",
            str(threads),
        ]
        if cpu_only:
            cmd.append("--cpu")

        log.info("Running SynthSeg QC on %d image(s).", len(input_paths))
        try:
            subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600 * max(1, len(input_paths)),
                env=os.environ.copy(),
                check=True,
            )
        except subprocess.CalledProcessError as e:
            if e.stderr:
                log.error("SynthSeg stderr:\n%s", e.stderr.strip())
            if e.stdout:
                log.error("SynthSeg stdout:\n%s", e.stdout.strip())
            raise


def run_preprocessing_pipeline(cfg: PipelineConfig) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    generated_root = cfg.output_dir / "generated"
    log_dir = cfg.output_dir / "logs" / "preprocessing"
    cmd = [
        "uv",
        "run",
        "python",
        str(repo_root / "src/preprocessing/pipeline.py"),
        "--input",
        str(generated_root),
        "--log-dir",
        str(log_dir),
        "--synthseg-threads",
        str(cfg.qc.threads),
    ]
    if cfg.qc.cpu:
        cmd.append("--cpu")
    log.info("Running preprocessing pipeline before SynthSeg QC.")
    subprocess.run(cmd, cwd=repo_root, check=True)


def attach_preprocessing_qc_paths(cfg: PipelineConfig, records: list[dict[str, Any]]) -> None:
    synthseg_dir = cfg.output_dir / "generated" / "derivatives" / "synthseg"
    for record in records:
        image_path = Path(record["image_path"])
        stem = image_path.name.removesuffix(".nii.gz")
        record["qc_path"] = str((synthseg_dir / f"{stem}_qc.csv").resolve())


def apply_qc_threshold(cfg: PipelineConfig, record: dict[str, Any]) -> None:
    qc_path = Path(record["qc_path"])
    try:
        scores = read_qc_scores(qc_path)
        qc_min = min(scores)
        qc_mean = sum(scores) / len(scores)
        metric_value = qc_min if cfg.qc.metric == "min" else qc_mean
        passed = True if cfg.qc.threshold is None else metric_value >= cfg.qc.threshold
        record["qc_min"] = qc_min
        record["qc_mean"] = qc_mean
        record["qc_metric"] = cfg.qc.metric
        record["qc_threshold"] = cfg.qc.threshold
        record["qc_pass"] = passed
        record["qc_error"] = None
    except Exception as e:
        record["qc_metric"] = cfg.qc.metric
        record["qc_threshold"] = cfg.qc.threshold
        record["qc_pass"] = False
        record["qc_error"] = str(e)
        raise


def read_qc_scores(path: Path) -> list[float]:
    if not path.exists():
        raise FileNotFoundError(f"QC file not found: {path}")
    with path.open(newline="") as f:
        sample = f.read(2048)
        f.seek(0)
        lines = sample.splitlines()
        if not lines:
            raise ValueError(f"QC file is empty: {path}")
        delimiter = "\t" if "\t" in lines[0] else ","
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"QC file is missing a header row: {path}")

        # SynthSeg's --qc CSV writes one image identifier column plus per-structure QC
        # scores, not one aggregate score column. Use every numeric cell as a QC score.
        scores = []
        for row in reader:
            for value in row.values():
                if value in {None, ""}:
                    continue
                try:
                    scores.append(float(value))
                except ValueError:
                    continue
    if not scores:
        raise ValueError(f"QC file has no readable QC score values: {path}")
    return scores
