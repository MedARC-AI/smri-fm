"""FOMO26 downstream eval tasks, built as HF datasets streamed from the challenge zips."""

import gzip
import os
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
from nibabel.processing import resample_from_to, resample_to_output

from nanobrain.eval.tasks import register_task
from nanobrain.eval.tasks.base import ClassificationTask, RegressionTask, SegmentationTask

BASE_URL = os.getenv("FOMO_EVAL_BASE_URL", "https://sid.erda.dk/share_redirect/fmeuvo1EdF")
TASK5_URL = os.getenv(
    "FOMO_EVAL_TASK5_URL",
    "https://huggingface.co/datasets/medarc/smri-fm/resolve/main/fomo_eval/Task_5.zip",
)

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


def _nifti(img: nib.Nifti1Image | bytes) -> dict:
    data = img if isinstance(img, bytes) else _dump(img)
    return {"path": None, "bytes": data}


def _load(nii_gz: bytes) -> nib.Nifti1Image:
    return nib.Nifti1Image.from_bytes(gzip.decompress(nii_gz))


def _dump(img: nib.Nifti1Image) -> bytes:
    return gzip.compress(img.to_bytes())


def _zero_seg_like(img: nib.Nifti1Image) -> nib.Nifti1Image:
    return nib.Nifti1Image(np.zeros(img.shape, dtype=np.uint8), img.affine)


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


def _resample(img: nib.Nifti1Image, min_spacing: Spacing, order: int) -> nib.Nifti1Image:
    """Resample to min_spacing, leaving any axis already coarser than it alone."""
    native = np.sqrt((img.affine[:3, :3] ** 2).sum(axis=0))
    target = tuple(np.maximum(native, min_spacing))
    return resample_to_output(img, voxel_sizes=target, order=order)


def _preprocess(
    image: nib.Nifti1Image,
    seg: nib.Nifti1Image | None = None,
    min_spacing: Spacing | None = MIN_SPACING,
    fov_mm: float | None = None,
) -> tuple[nib.Nifti1Image, nib.Nifti1Image | None]:
    """Resample an image and land its segmentation on exactly that grid, then center-crop both.

    The seg is resampled onto the image's output grid rather than resampled independently:
    some tasks store segs whose affine differs from the image's by float noise, which is
    enough to round to an off-by-one output shape and break the probe's grid contract.
    """
    if min_spacing is not None:
        image = _resample(image, min_spacing, order=1)
    if seg is not None:
        seg = resample_from_to(seg, (image.shape, image.affine), order=0)
    if fov_mm is not None:
        image = _center_crop(image, fov_mm)
        if seg is not None:
            seg = _center_crop(seg, fov_mm)
    return image, seg


def _from_generator(generator, features: Features, **gen_kwargs) -> Dataset:
    # Small batches keep each Arrow binary chunk under the 2 GB offset limit for raw niftis.
    return Dataset.from_generator(
        generator, features=features, gen_kwargs=gen_kwargs, writer_batch_size=16
    )


# ---- Task 1: acute infarct (classification; positives also carry a lesion mask) --------


def _generate_task1(min_spacing: Spacing) -> Iterator[dict]:
    url = f"{BASE_URL}/Task_1.zip"
    with _open_zip(url) as zf:
        names = set(zf.namelist())
        for sub in _subjects(zf, "dwi_b1000.nii.gz"):
            stem = f"Task_1/{{}}/{sub}/ses-01"
            label = int(zf.read(stem.format("labels") + "/label.txt").strip())
            image = _load(zf.read(stem.format("preprocessed") + "/dwi_b1000.nii.gz"))
            seg_name = stem.format("labels") + "/seg.nii.gz"
            seg = _load(zf.read(seg_name)) if seg_name in names else _zero_seg_like(image)
            image, seg = _preprocess(image, seg, min_spacing)
            yield {"subject": sub, "label": label, "image": _nifti(image), "seg": _nifti(seg)}


def load_task1() -> Dataset:
    features = Features(
        {"subject": Value("string"), "label": Value("int32"), "image": Nifti(), "seg": Nifti()}
    )
    return _from_generator(_generate_task1, features, min_spacing=MIN_SPACING)


@register_task
def fomo_task1_infarct() -> ClassificationTask:
    return ClassificationTask(name="fomo_task1_infarct", dataset_fn=load_task1, target_col="label")


@register_task
def fomo_task1_infarct_seg() -> SegmentationTask:
    return SegmentationTask(
        name="fomo_task1_infarct_seg",
        dataset_fn=load_task1,
        seg_col="seg",
        class_names=("infarct",),
    )


# ---- Task 2: meningioma segmentation ---------------------------------------------------


def _generate_task2(min_spacing: Spacing) -> Iterator[dict]:
    url = f"{BASE_URL}/Task_2.zip"
    with _open_zip(url) as zf:
        for sub in _subjects(zf, "seg.nii.gz"):
            stem = f"Task_2/{{}}/{sub}/ses-01"
            image = _load(zf.read(stem.format("preprocessed") + "/flair.nii.gz"))
            seg = _load(zf.read(stem.format("labels") + "/seg.nii.gz"))
            image, seg = _preprocess(image, seg, min_spacing)
            yield {"subject": sub, "image": _nifti(image), "seg": _nifti(seg)}


def load_task2() -> Dataset:
    features = Features({"subject": Value("string"), "image": Nifti(), "seg": Nifti()})
    return _from_generator(_generate_task2, features, min_spacing=MIN_SPACING)


@register_task
def fomo_task2_meningioma() -> SegmentationTask:
    return SegmentationTask(
        name="fomo_task2_meningioma",
        dataset_fn=load_task2,
        seg_col="seg",
        class_names=("meningioma",),
    )


# ---- Task 3: brain age regression ------------------------------------------------------


def _generate_task3() -> Iterator[dict]:
    url = f"{BASE_URL}/Task_3.zip"
    with _open_zip(url) as zf:
        for sub in _subjects(zf, "t1w.nii.gz"):
            age = int(zf.read(f"Task_3/labels/{sub}/ses-01/labels.txt").strip())
            image = _load(zf.read(f"Task_3/preprocessed/{sub}/ses-01/t1w.nii.gz"))
            yield {"subject": sub, "age": age, "image": _nifti(image)}


def load_task3() -> Dataset:
    features = Features({"subject": Value("string"), "age": Value("int32"), "image": Nifti()})
    return _from_generator(_generate_task3, features)


@register_task
def fomo_task3_age() -> RegressionTask:
    return RegressionTask(name="fomo_task3_age", dataset_fn=load_task3, target_col="age")


# ---- Task 4: trigeminal nerve/vessel segmentation --------------------------------------


def _generate_task4(min_spacing: Spacing, fov_mm: float) -> Iterator[dict]:
    url = f"{BASE_URL}/Task_4.zip"
    with _open_zip(url) as zf:
        for sub in _subjects(zf, "seg.nii.gz"):
            stem = f"Task_4/{{}}/{sub}/ses-01"
            image = _load(zf.read(stem.format("preprocessed") + "/t2w.nii.gz"))
            seg = _load(zf.read(stem.format("labels") + "/seg.nii.gz"))
            image, seg = _preprocess(image, seg, min_spacing, fov_mm)
            yield {"subject": sub, "image": _nifti(image), "seg": _nifti(seg)}


def load_task4() -> Dataset:
    features = Features({"subject": Value("string"), "image": Nifti(), "seg": Nifti()})
    return _from_generator(
        _generate_task4, features, min_spacing=TASK4_SPACING, fov_mm=TASK4_FOV_MM
    )


@register_task
def fomo_task4_trigeminal() -> SegmentationTask:
    return SegmentationTask(
        name="fomo_task4_trigeminal",
        dataset_fn=load_task4,
        seg_col="seg",
        # TODO: confirm label order in the challenge data (label 1 -> nerve, label 2 -> vessel).
        class_names=("nerve", "vessel"),
    )


# ---- Task 5: polymicrogyria classification ---------------------------------------------


def _generate_task5() -> Iterator[dict]:
    with _open_zip(TASK5_URL) as zf:
        for sub in _subjects(zf, "t1.nii.gz"):
            label = int(zf.read(f"Task_5/labels/{sub}/ses_01/labels.txt").strip())
            image_gz = zf.read(f"Task_5/preprocessed/{sub}/ses_01/t1.nii.gz")
            yield {"subject": sub, "label": label, "image": _nifti(image_gz)}


def load_task5() -> Dataset:
    features = Features({"subject": Value("string"), "label": Value("int32"), "image": Nifti()})
    return _from_generator(_generate_task5, features)


@register_task
def fomo_task5_polymicrogyria() -> ClassificationTask:
    return ClassificationTask(
        name="fomo_task5_polymicrogyria", dataset_fn=load_task5, target_col="label"
    )
