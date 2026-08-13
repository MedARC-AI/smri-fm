"""How far does the shipped task 3 head move when the input is perturbed the way another site's
data would differ from this one?

Cross-validation says r=0.963 on the challenge's own 494; the validation leaderboard says 0.426.
Nothing in the protocol, the cohort or the container accounts for that, so the remaining
explanation is that the held-out cohort does not look like this one. This puts a number on it:
each perturbation is a difference a different scanner or a different preprocessing pipeline could
plausibly introduce, and the cost is reported in years of predicted age.

The head is the one that was submitted, applied to features re-extracted from the perturbed image,
so `identity` reproduces the shipped in-sample prediction exactly and is the control.

Needs a GPU. ~15 minutes for 100 subjects x 14 conditions.
"""

from multiprocessing import Pool
from pathlib import Path

import joblib
import nibabel as nib
import numpy as np
import torch
from omegaconf import OmegaConf
from scipy import ndimage

from fomo_tune.main_task3 import Config, Task3Method

ROOT = Path(__file__).parents[2]
TASK_DIR = ROOT / "data/fomo_eval/Task_3"
RUN_DIR = ROOT / "experiments/fomo_tune_baseline/output/task3"
OUT_DIR = Path(__file__).parent
N_SUBJECTS = 100


def brain_mask_op(data: np.ndarray, iterations: int) -> np.ndarray:
    """A tighter or looser skull strip, which is where two pipelines most visibly disagree."""
    brain = data > 0
    if iterations > 0:
        grown = ndimage.binary_dilation(brain, iterations=iterations)
        # dilation has no intensities to grow into, so fill with a local blur of the edge
        filled = ndimage.gaussian_filter(data, 1.0)
        return np.where(grown & ~brain, filled, data)
    return np.where(ndimage.binary_erosion(brain, iterations=-iterations), data, 0.0)


def ramp(shape: tuple[int, ...], strength: float) -> np.ndarray:
    """A smooth multiplicative gradient along y, standing in for residual bias field."""
    y = np.linspace(1 - strength, 1 + strength, shape[1], dtype=np.float32)
    return y[None, :, None]


def acquired_at(data: np.ndarray, mm: float) -> np.ndarray:
    """Detail a thicker acquisition would not have resolved, at the same physical size."""
    coarse = ndimage.zoom(data, 1 / mm, order=1)
    return ndimage.zoom(coarse, np.array(data.shape) / np.array(coarse.shape), order=1)


def background_noise(data: np.ndarray, rng: np.random.Generator, fraction: float) -> np.ndarray:
    """Rectified gaussian outside the brain: what a scan looks like if nothing zeroed the air."""
    level = fraction * np.median(data[data > 0])
    noise = np.abs(rng.normal(0, level, data.shape)).astype(np.float32)
    return np.where(data > 0, data, noise)


# `voxel_size_*` are applied to the affine rather than the array, further down: the array is
# unchanged and the header claims larger voxels, so the head arrives at the backbone scaled up.
PERTURBATIONS = {
    "identity": lambda d, rng: d,
    "shift_x_10mm": lambda d, rng: ndimage.shift(d, (10, 0, 0), order=1),
    "shift_z_10mm": lambda d, rng: ndimage.shift(d, (0, 0, 10), order=1),
    "rotate_10deg": lambda d, rng: ndimage.rotate(d, 10, axes=(0, 1), reshape=False, order=1),
    "flip_lr": lambda d, rng: d[::-1].copy(),
    "intensity_x1.25": lambda d, rng: d * 1.25,
    "bias_field_10pct": lambda d, rng: d * ramp(d.shape, 0.10),
    "blur_1mm": lambda d, rng: ndimage.gaussian_filter(d, 1.0),
    "blur_2mm": lambda d, rng: ndimage.gaussian_filter(d, 2.0),
    "acquired_at_1.5mm": lambda d, rng: acquired_at(d, 1.5),
    "acquired_at_2mm": lambda d, rng: acquired_at(d, 2.0),
    "noise_in_brain_5pct": lambda d, rng: np.where(
        d > 0, d + rng.normal(0, 0.05 * np.median(d[d > 0]), d.shape).astype(np.float32), d
    ),
    "background_noise_2pct": lambda d, rng: background_noise(d, rng, 0.02),
    "background_noise_5pct": lambda d, rng: background_noise(d, rng, 0.05),
    "strip_tighter_2mm": lambda d, rng: brain_mask_op(d, -2),
    "strip_looser_2mm": lambda d, rng: brain_mask_op(d, 2),
    "voxel_size_1.05x": lambda d, rng: d,
    "voxel_size_1.10x": lambda d, rng: d,
    "voxel_size_1.20x": lambda d, rng: d,
    "voxel_size_0.90x": lambda d, rng: d,
}


def apply_one(name: str, data: np.ndarray, seed: int) -> tuple[str, np.ndarray]:
    """Run in a worker: the perturbations are scipy on 11.5M voxels and dominate the wall clock."""
    return name, np.ascontiguousarray(PERTURBATIONS[name](data, np.random.default_rng(seed)))


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.structured(Config), OmegaConf.load(RUN_DIR / "config.yaml"))
    method = Task3Method(cfg)
    method.head = joblib.load(RUN_DIR / "model/head.joblib")

    saved = np.load(OUT_DIR / "oof.npz")
    subjects, ages = list(saved["subjects"]), saved["age"]
    # spread over the age range rather than the file order, which is grouped by source cohort
    keep = np.argsort(ages)[np.linspace(0, len(subjects) - 1, N_SUBJECTS).astype(int)]

    predictions = {name: [] for name in PERTURBATIONS}
    with Pool(len(PERTURBATIONS)) as pool:
        for count, i in enumerate(keep):
            path = TASK_DIR / f"preprocessed/{subjects[i]}/ses-01/t1w.nii.gz"
            img = nib.load(path)
            data = np.asarray(img.dataobj, dtype=np.float32)

            outputs = dict(pool.starmap(apply_one, [(n, data, i) for n in PERTURBATIONS]))
            for name in PERTURBATIONS:
                affine = img.affine.copy()
                if name.startswith("voxel_size_"):
                    affine[:3, :3] *= float(name.removeprefix("voxel_size_").removesuffix("x"))
                perturbed = nib.Nifti1Image(outputs[name], affine)
                predictions[name].append(method.predict({"t1w": perturbed}))
            print(f"{count + 1}/{len(keep)} {subjects[i]}", flush=True)

    np.savez(
        OUT_DIR / "perturb.npz",
        subjects=np.array([subjects[i] for i in keep]),
        age=ages[keep],
        **{name: np.array(v) for name, v in predictions.items()},
    )

    base = np.array(predictions["identity"])
    truth = ages[keep].astype(float)
    print(f"\n{'perturbation':24s} {'r':>7s} {'mae':>7s} {'d(pred) mean':>13s} {'|d(pred)|':>10s}")
    for name, values in predictions.items():
        v = np.array(values)
        print(
            f"{name:24s} {np.corrcoef(truth, v)[0, 1]:7.3f} {np.abs(truth - v).mean():7.2f} "
            f"{np.mean(v - base):13.2f} {np.abs(v - base).mean():10.2f}"
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
