import argparse
import logging
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import ants
import nibabel as nib
import numpy as np

log = logging.getLogger("preprocess")

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
DEFAULT_SYNTHSEG_CMD = (
    "uvx --python 3.11 --from 'git+https://github.com/MedARC-AI/SynthSeg.git' SynthSeg"
)
TEMPLATE_SPACE = "MNI152NLin2009cAsym"


def setup_logging(log_file: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        force=True,
    )


def scan_stem(input_path: Path) -> str:
    return input_path.name.removesuffix(".nii.gz")


def output_paths(input_path: Path, input_dir: Path) -> tuple[Path, Path, Path]:
    stem = input_path.name.removesuffix(".nii.gz")
    processed_dir = input_dir / "processed"
    mask_dir = input_dir / "derivatives" / "masks"
    xfm_dir = input_dir / "derivatives" / "transforms"
    for d in (processed_dir, mask_dir, xfm_dir):
        d.mkdir(parents=True, exist_ok=True)
    processed = processed_dir / f"{stem}_space-{TEMPLATE_SPACE}_desc-processed.nii.gz"
    mask = mask_dir / f"{stem}_space-{TEMPLATE_SPACE}_desc-brain_mask.nii.gz"
    xfm = xfm_dir / f"{stem}_from-native_to-{TEMPLATE_SPACE}_mode-image_xfm.mat"
    return processed, mask, xfm


def synthseg_output_paths(input_path: Path, input_dir: Path) -> tuple[Path, Path, Path]:
    stem = scan_stem(input_path)
    d = input_dir / "derivatives" / "synthseg"
    d.mkdir(parents=True, exist_ok=True)
    seg = d / f"{stem}_desc-synthseg_dseg.nii.gz"
    vol = d / f"{stem}_volumes.csv"
    qc = d / f"{stem}_qc.csv"
    return seg, vol, qc


def nib_to_ants(img: nib.Nifti1Image):
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
        nib.save(img, tmp.name)
        ants_img = ants.image_read(tmp.name)
    Path(tmp.name).unlink(missing_ok=True)
    return ants_img


def ants_to_nib(ants_img) -> nib.Nifti1Image:
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
        ants.image_write(ants_img, tmp.name)
        img = nib.load(tmp.name)
        img = nib.Nifti1Image(img.get_fdata(), img.affine, img.header)
    Path(tmp.name).unlink(missing_ok=True)
    return img


def save_brain_mask_from_segmentation(seg_path: Path, mask_path: Path) -> None:
    seg_img = nib.load(seg_path)
    mask = (np.asanyarray(seg_img.dataobj) > 0).astype(np.uint8)
    header = seg_img.header.copy()
    header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(mask, seg_img.affine, header), mask_path)


def rigid_register_to_template(
    img: nib.Nifti1Image,
    template_brain_path: Path,
    transform_path: Path,
) -> nib.Nifti1Image:
    fixed = ants.image_read(str(template_brain_path))
    moving = nib_to_ants(img)

    with tempfile.TemporaryDirectory() as tmpdir:
        outprefix = str(Path(tmpdir) / "rigid_")
        tx = ants.registration(
            fixed=fixed,
            moving=moving,
            type_of_transform="Rigid",
            outprefix=outprefix,
        )
        fwdtransforms = tx["fwdtransforms"]
        rigid_transform = next(
            (Path(path) for path in fwdtransforms if str(path).endswith(".mat")), None
        )
        if rigid_transform is None:
            raise RuntimeError("ANTs rigid registration did not produce a matrix transform")

        registered = ants.apply_transforms(
            fixed=fixed,
            moving=moving,
            transformlist=fwdtransforms,
            interpolator="bSpline",
        )
        shutil.copy2(rigid_transform, transform_path)

    return ants_to_nib(registered)


def _default_template_brain() -> Path:
    import templateflow.api as tflow
    return Path(str(tflow.get(
        TEMPLATE_SPACE, resolution=1, desc="brain", suffix="T1w", extension=".nii.gz"
    )))


def run_synthseg(
    input_paths: list[Path],
    seg_paths: list[Path],
    vol_paths: list[Path],
    qc_paths: list[Path],
    synthseg_cmd: str,
    threads: int,
    cpu_only: bool,
    timeout: float = 600,
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
            "--i", str(input_txt),
            "--o", str(output_txt),
            "--vol", str(vol_txt),
            "--qc", str(qc_txt),
            "--parc",
            "--robust",
            "--threads", str(threads),
        ]
        if cpu_only:
            cmd.append("--cpu")
        subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=os.environ.copy(),
            check=True,
        )


def process_file(input_path: Path, input_dir: Path, template_brain_path: Path) -> bool:
    name = str(input_path.relative_to(input_dir))
    processed_path, _, xfm_path = output_paths(input_path, input_dir)

    if processed_path.exists() and xfm_path.exists():
        log.info("%s — already registered, skipping", name)
        return True

    log.info("%s — rigid registration to %s", name, TEMPLATE_SPACE)
    try:
        img = nib.load(input_path)
        registered = rigid_register_to_template(img, template_brain_path, xfm_path)
        nib.save(registered, processed_path)
        log.info("%s — done → %s", name, processed_path.name)
        return True
    except Exception as e:
        log.error("%s — failed: %s", name, e, exc_info=True)
        return False


def process_synthseg_batch(
    tasks: list[tuple[Path, Path, Path, Path, Path, Path]],
    synthseg_cmd: str,
    threads: int,
    cpu_only: bool,
) -> list[str]:
    failed: list[str] = []
    pending: list[tuple] = []

    for orig_path, processed_path, seg_path, vol_path, qc_path, mask_path in tasks:
        name = orig_path.name
        if seg_path.exists() and vol_path.exists() and qc_path.exists():
            log.info("%s — SynthSeg already done, skipping", name)
            if not mask_path.exists():
                try:
                    save_brain_mask_from_segmentation(seg_path, mask_path)
                    log.info("%s — brain mask done → %s", name, mask_path.name)
                except Exception as e:
                    log.error("%s — brain mask failed: %s", name, e, exc_info=True)
                    failed.append(name)
        else:
            pending.append((name, processed_path, seg_path, vol_path, qc_path, mask_path))

    if not pending:
        return failed

    try:
        inputs = [processed_path for _, processed_path, *_ in pending]
        segs = [seg_path for _, _, seg_path, *_ in pending]
        vols = [vol_path for _, _, _, vol_path, *_ in pending]
        qcs = [qc_path for _, _, _, _, qc_path, *_ in pending]

        log.info("SynthSeg: running on %d scan(s)", len(pending))
        run_synthseg(
            inputs,
            segs,
            vols,
            qcs,
            synthseg_cmd,
            threads,
            cpu_only,
            timeout=600 * len(pending),
        )

        for name, _, seg_path, vol_path, qc_path, mask_path in pending:
            log.info("%s — SynthSeg done → %s", name, seg_path.name)
            log.info("%s — SynthSeg sidecars → %s, %s", name, vol_path.name, qc_path.name)
            save_brain_mask_from_segmentation(seg_path, mask_path)
            log.info("%s — brain mask done → %s", name, mask_path.name)

    except subprocess.CalledProcessError as e:
        log.error("SynthSeg batch failed: %s", e, exc_info=True)
        for label, text in (("stderr", e.stderr), ("stdout", e.output)):
            if text := (text or "").strip():
                log.error("SynthSeg %s:\n%s", label, text)
        for name, *_ in pending:
            failed.append(name)
    except Exception as e:
        log.error("SynthSeg batch failed: %s", e, exc_info=True)
        for name, *_ in pending:
            failed.append(name)

    return failed


def main() -> None:
    parser = argparse.ArgumentParser(description="Anat preprocessing pipeline")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--log-file", type=Path)
    parser.add_argument(
        "--template-brain", default=None, type=Path, dest="template_brain",
        help="Template brain image for rigid registration (defaults to MNI152NLin2009cAsym res-1 via templateflow).",
    )
    parser.add_argument("--itk-threads", default=2, type=int)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--synthseg-threads", default=8, type=int)
    parser.add_argument("--batch-id", default=0, type=int)
    parser.add_argument("--batch-size", default=None, type=int)
    args = parser.parse_args()

    input_dir = args.input.resolve()
    log_dir = (args.log_dir or input_dir / "logs").resolve()
    template_brain = (args.template_brain or _default_template_brain()).resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not template_brain.exists():
        raise FileNotFoundError(f"Template brain not found: {template_brain}")
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = (args.log_file.resolve() if args.log_file else log_dir / "run.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    setup_logging(log_file)
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(args.itk_threads)

    log.info("Input: %s", input_dir)
    log.info("Processed: %s", input_dir / "processed")
    log.info("SynthSeg: %s", input_dir / "derivatives" / "synthseg")
    log.info("Template brain: %s", template_brain)
    log.info("SynthSeg command: %s", DEFAULT_SYNTHSEG_CMD)

    excluded = {input_dir / "processed", input_dir / "derivatives", input_dir / "logs"}
    files = sorted(
        f for f in input_dir.rglob("*.nii.gz")
        if not any(f.is_relative_to(d) for d in excluded)
    )
    if not files:
        log.info("No .nii.gz files found in %s", input_dir)
        return

    if args.batch_size is not None:
        start = args.batch_id * args.batch_size
        stop = start + args.batch_size
        files = files[start:stop]

    log.info(
        "Processing %d file(s) (batch_id=%d, batch_size=%s)",
        len(files), args.batch_id, args.batch_size,
    )

    if not files:
        log.info("No files in this batch.")
        return

    reg_failed = []
    synthseg_tasks = []
    for f in files:
        if process_file(f, input_dir, template_brain):
            processed_path, mask_path, _ = output_paths(f, input_dir)
            seg_path, vol_path, qc_path = synthseg_output_paths(f, input_dir)
            synthseg_tasks.append((f, processed_path, seg_path, vol_path, qc_path, mask_path))
        else:
            reg_failed.append(str(f.relative_to(input_dir)))

    if reg_failed:
        log.error("Registration failed for: %s", reg_failed)

    ss_failed = process_synthseg_batch(
        synthseg_tasks, DEFAULT_SYNTHSEG_CMD, args.synthseg_threads, args.cpu
    )
    if ss_failed:
        log.error("SynthSeg failed for: %s", ss_failed)
    if reg_failed or ss_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
