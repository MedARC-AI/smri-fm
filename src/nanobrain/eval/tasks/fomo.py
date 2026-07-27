"""FOMO26 downstream eval tasks, built as HF datasets streamed from the challenge zips.

Each generator reads a task zip (remote URL by default, or a local path for testing) and
yields per-subject samples without a manual download step.
"""

import gzip
import shutil
import tempfile
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import fsspec
import nibabel as nib
import numpy as np
from datasets import Dataset, Features, Nifti, Value
from nibabel.processing import resample_to_output

from nanobrain.eval.tasks import register_task
from nanobrain.eval.tasks.base import ClassificationTask, RegressionTask, SegmentationTask

BASE_URL = "https://sid.erda.dk/share_redirect/fmeuvo1EdF"

Spacing = tuple[float, float, float]

MIN_SPACING: Spacing = (1.0, 1.0, 1.0)
TASK4_SPACING: Spacing = (0.5, 0.5, 0.5)
TASK4_FOV_MM = 100.0


@contextmanager
def _open_zip(url: str) -> Iterator[zipfile.ZipFile]:
    """Open a task zip, copying a remote url to a temp file first."""
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(url)
        if not local.exists():
            local = Path(tmp) / "task.zip"
            with fsspec.open(url) as src, local.open("wb") as dst:
                shutil.copyfileobj(src, dst)
        with zipfile.ZipFile(local) as zf:
            yield zf


def _subjects(zf: zipfile.ZipFile, suffix: str) -> list[str]:
    return sorted({name.split("/")[2] for name in zf.namelist() if name.endswith(suffix)})


def _nifti(data: bytes) -> dict:
    return {"path": None, "bytes": data}


def _zero_seg_like(image_gz: bytes) -> bytes:
    """An all-background .nii.gz segmentation matching an image's grid (label-negative subjects)."""
    img = nib.Nifti1Image.from_bytes(gzip.decompress(image_gz))
    zeros = nib.Nifti1Image(np.zeros(img.shape, dtype=np.uint8), img.affine)
    return gzip.compress(zeros.to_bytes())


def _center_crop(img: nib.Nifti1Image, fov_mm: float) -> nib.Nifti1Image:
    """Center-crop to an fov_mm cube."""
    spacing = np.sqrt((img.affine[:3, :3] ** 2).sum(axis=0))
    size = np.round(fov_mm / spacing).astype(int)
    start = (np.array(img.shape) - size) // 2
    assert (start >= 0).all(), f"fov {fov_mm}mm exceeds volume {img.shape}"
    return img.slicer[
        start[0] : start[0] + size[0],
        start[1] : start[1] + size[1],
        start[2] : start[2] + size[2],
    ]


def _preprocess_bytes(
    nii_gz: bytes, min_spacing: Spacing, order: int, fov_mm: float | None = None
) -> bytes:
    """Resample a .nii.gz and center-crop."""
    img = nib.Nifti1Image.from_bytes(gzip.decompress(nii_gz))
    native = np.sqrt((img.affine[:3, :3] ** 2).sum(axis=0))
    target = np.maximum(native, min_spacing)
    out = resample_to_output(img, voxel_sizes=tuple(target), order=order)
    if fov_mm is not None:
        out = _center_crop(out, fov_mm)
    return gzip.compress(out.to_bytes())


def _from_generator(generator, features: Features, **gen_kwargs) -> Dataset:
    # Small batches keep each Arrow binary chunk under the 2 GB offset limit for raw niftis.
    return Dataset.from_generator(
        generator, features=features, gen_kwargs=gen_kwargs, writer_batch_size=16
    )


# ---- Task 1: acute infarct (classification; positives also carry a lesion mask) --------


def _generate_task1(url: str, min_spacing: Spacing | None) -> Iterator[dict]:
    with _open_zip(url) as zf:
        names = set(zf.namelist())
        for sub in _subjects(zf, "dwi_b1000.nii.gz"):
            stem = f"Task_1/{{}}/{sub}/ses-01"
            label = int(zf.read(stem.format("labels") + "/label.txt").strip())
            image = zf.read(stem.format("preprocessed") + "/dwi_b1000.nii.gz")
            seg_name = stem.format("labels") + "/seg.nii.gz"
            seg = zf.read(seg_name) if seg_name in names else _zero_seg_like(image)
            if min_spacing is not None:
                image = _preprocess_bytes(image, min_spacing, order=1)
                seg = _preprocess_bytes(seg, min_spacing, order=0)
            yield {"subject": sub, "label": label, "image": _nifti(image), "seg": _nifti(seg)}


def load_task1(url: str, min_spacing: Spacing | None = None) -> Dataset:
    features = Features(
        {"subject": Value("string"), "label": Value("int32"), "image": Nifti(), "seg": Nifti()}
    )
    return _from_generator(_generate_task1, features, url=url, min_spacing=min_spacing)


@register_task
def fomo_task1_infarct(url: str = f"{BASE_URL}/Task_1.zip") -> ClassificationTask:
    return ClassificationTask(
        name="fomo_task1_infarct", dataset_fn=lambda: load_task1(url), target_col="label"
    )


@register_task
def fomo_task1_infarct_seg(url: str = f"{BASE_URL}/Task_1.zip") -> SegmentationTask:
    return SegmentationTask(
        name="fomo_task1_infarct_seg",
        dataset_fn=lambda: load_task1(url, MIN_SPACING),
        seg_col="seg",
        class_names=("infarct",),
    )


# ---- Task 2: meningioma segmentation ---------------------------------------------------


def _generate_task2(url: str, min_spacing: Spacing) -> Iterator[dict]:
    with _open_zip(url) as zf:
        for sub in _subjects(zf, "seg.nii.gz"):
            stem = f"Task_2/{{}}/{sub}/ses-01"
            image = _preprocess_bytes(
                zf.read(stem.format("preprocessed") + "/flair.nii.gz"), min_spacing, 1
            )
            seg = _preprocess_bytes(zf.read(stem.format("labels") + "/seg.nii.gz"), min_spacing, 0)
            yield {"subject": sub, "image": _nifti(image), "seg": _nifti(seg)}


def load_task2(url: str, min_spacing: Spacing = MIN_SPACING) -> Dataset:
    features = Features({"subject": Value("string"), "image": Nifti(), "seg": Nifti()})
    return _from_generator(_generate_task2, features, url=url, min_spacing=min_spacing)


@register_task
def fomo_task2_meningioma(url: str = f"{BASE_URL}/Task_2.zip") -> SegmentationTask:
    return SegmentationTask(
        name="fomo_task2_meningioma",
        dataset_fn=lambda: load_task2(url),
        seg_col="seg",
        class_names=("meningioma",),
    )


# ---- Task 3: brain age regression ------------------------------------------------------


def _generate_task3(url: str) -> Iterator[dict]:
    with _open_zip(url) as zf:
        for sub in _subjects(zf, "t1w.nii.gz"):
            age = int(zf.read(f"Task_3/labels/{sub}/ses-01/labels.txt").strip())
            image = zf.read(f"Task_3/preprocessed/{sub}/ses-01/t1w.nii.gz")
            yield {"subject": sub, "age": age, "image": _nifti(image)}


def load_task3(url: str) -> Dataset:
    features = Features({"subject": Value("string"), "age": Value("int32"), "image": Nifti()})
    return _from_generator(_generate_task3, features, url=url)


@register_task
def fomo_task3_age(url: str = f"{BASE_URL}/Task_3.zip") -> RegressionTask:
    return RegressionTask(
        name="fomo_task3_age", dataset_fn=lambda: load_task3(url), target_col="age"
    )


# ---- Task 4: trigeminal nerve/vessel segmentation --------------------------------------


def _generate_task4(url: str, min_spacing: Spacing, fov_mm: float) -> Iterator[dict]:
    with _open_zip(url) as zf:
        for sub in _subjects(zf, "seg.nii.gz"):
            stem = f"Task_4/{{}}/{sub}/ses-01"
            image = _preprocess_bytes(
                zf.read(stem.format("preprocessed") + "/t2w.nii.gz"), min_spacing, 1, fov_mm
            )
            seg = _preprocess_bytes(
                zf.read(stem.format("labels") + "/seg.nii.gz"), min_spacing, 0, fov_mm
            )
            yield {"subject": sub, "image": _nifti(image), "seg": _nifti(seg)}


def load_task4(
    url: str, min_spacing: Spacing = TASK4_SPACING, fov_mm: float = TASK4_FOV_MM
) -> Dataset:
    features = Features({"subject": Value("string"), "image": Nifti(), "seg": Nifti()})
    return _from_generator(
        _generate_task4, features, url=url, min_spacing=min_spacing, fov_mm=fov_mm
    )


@register_task
def fomo_task4_trigeminal(url: str = f"{BASE_URL}/Task_4.zip") -> SegmentationTask:
    return SegmentationTask(
        name="fomo_task4_trigeminal",
        dataset_fn=lambda: load_task4(url),
        seg_col="seg",
        # TODO: confirm label order in the challenge data (label 1 -> nerve, label 2 -> vessel).
        class_names=("nerve", "vessel"),
    )


# ---- Task 5: polymicrogyria classification ---------------------------------------------

# TODO: back up Task_5 to online location
TASK5_DIR = "data/fomo_eval/Task_5"


def _generate_task5(root: str) -> Iterator[dict]:
    base = Path(root)
    for sub_dir in sorted((base / "preprocessed").iterdir()):
        sub = sub_dir.name
        label = int((base / "labels" / sub / "ses_01" / "labels.txt").read_text().strip())
        image = (sub_dir / "ses_01" / "t1.nii.gz").read_bytes()
        yield {"subject": sub, "label": label, "image": _nifti(image)}


def load_task5(root: str) -> Dataset:
    features = Features({"subject": Value("string"), "label": Value("int32"), "image": Nifti()})
    return Dataset.from_generator(
        _generate_task5, features=features, gen_kwargs={"root": root}, writer_batch_size=16
    )


@register_task
def fomo_task5_polymicrogyria(root: str = TASK5_DIR) -> ClassificationTask:
    return ClassificationTask(
        name="fomo_task5_polymicrogyria", dataset_fn=lambda: load_task5(root), target_col="label"
    )
