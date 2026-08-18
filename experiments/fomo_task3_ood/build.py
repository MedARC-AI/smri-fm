"""Build the fixed augmented DLBS external test for FOMO Task 3."""

import argparse
import json
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import ndimage

SEED = 4466


def resample_acquisition(
    data: np.ndarray,
    mask: np.ndarray,
    affine: np.ndarray,
    target_spacing: np.ndarray,
    profile: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    old_shape = np.asarray(data.shape)
    spacing = nib.affines.voxel_sizes(affine)
    if profile == "gaussian":
        added_fwhm = np.sqrt(np.maximum(target_spacing**2 - spacing**2, 0))
        data = ndimage.gaussian_filter(data, added_fwhm / (2.355 * spacing))
    else:
        assert profile == "boxcar"
        for axis, width in enumerate(np.maximum(1, np.rint(target_spacing / spacing)).astype(int)):
            data = ndimage.uniform_filter1d(data, width, axis=axis, mode="nearest")
    new_shape = np.maximum(1, np.rint(old_shape * spacing / target_spacing)).astype(int)
    data = (
        F.interpolate(
            torch.from_numpy(data)[None, None],
            size=tuple(new_shape),
            mode="trilinear",
            align_corners=False,
        )
        .squeeze()
        .numpy()
    )
    mask = (
        F.interpolate(
            torch.from_numpy(mask.astype(np.float32))[None, None],
            size=tuple(new_shape),
            mode="nearest-exact",
        )
        .squeeze()
        .bool()
        .numpy()
    )
    scale = old_shape / new_shape
    step = np.diag([*scale, 1.0])
    step[:3, 3] = 0.5 * scale - 0.5
    return data, mask, affine @ step


def augment_subject(job: tuple[str, int, str, str]) -> dict:
    torch.set_num_threads(1)
    subject, age, input_path, output_path = job
    image = nib.load(input_path)
    data = image.get_fdata(dtype=np.float32)
    mask = data > 0
    affine = image.affine.copy()
    rng = np.random.default_rng(np.random.SeedSequence([SEED, sum(map(ord, subject)), 13]))

    strength = 1.4
    angles = rng.uniform(-8 * strength, 8 * strength, 3)
    x, y, z = np.deg2rad(angles)
    rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
    scale = rng.uniform(1 - 0.10 * strength, 1 + 0.10 * strength)
    inverse = np.linalg.inv(scale * (rz @ ry @ rx))
    shift = rng.uniform(-8 * strength, 8 * strength, 3)
    centre = (np.asarray(data.shape) - 1) / 2
    offset = centre - inverse @ (centre + shift)
    data = ndimage.affine_transform(data, inverse, offset=offset, order=1)
    mask = ndimage.affine_transform(mask, inverse, offset=offset, order=0) > 0

    brain = data[mask]
    low, high = np.percentile(brain, (1, 99))
    gamma = rng.uniform(1 - 0.35 * 1.1, 1 + 0.45 * 1.1)
    data = low + (high - low) * np.clip((data - low) / (high - low), 0, 1) ** gamma
    bias = rng.uniform(-0.25 * 1.1, 0.25 * 1.1, 3)
    coords = np.meshgrid(
        *[np.linspace(-1, 1, n, dtype=np.float32) for n in data.shape], indexing="ij"
    )
    data *= np.clip(1 + sum(coef * coord for coef, coord in zip(bias, coords)), 0.5, 1.5)
    noise_sigma = 0.025 * 1.1 * (high - low)
    n1 = rng.normal(0, noise_sigma, data.shape).astype(np.float32)
    n2 = rng.normal(0, noise_sigma, data.shape).astype(np.float32)
    data = np.sqrt((np.maximum(data, 0) + n1) ** 2 + n2**2)

    ghost_axis = int(rng.integers(0, 3))
    ghost_shift = np.zeros(3)
    ghost_shift[ghost_axis] = rng.uniform(6, 12)
    ghost_weight = rng.uniform(0.18, 0.28)
    data = (1 - ghost_weight) * data + ghost_weight * ndimage.shift(data, ghost_shift, order=1)
    erosion = int(rng.integers(1, 4))
    mask = ndimage.binary_erosion(mask, iterations=erosion)

    family = ("anisotropic", "isotropic", "reconstruction")[int(rng.integers(0, 3))]
    if family == "anisotropic":
        axis = int(rng.integers(0, 3))
        target = nib.affines.voxel_sizes(affine).copy()
        target[axis] = rng.uniform(4, 7)
        data, mask, affine = resample_acquisition(data, mask, affine, target, "boxcar")
        acquisition = {"axis": axis, "spacing_mm": target.tolist(), "profile": "boxcar"}
    elif family == "isotropic":
        target = np.full(3, rng.uniform(2, 3.2))
        data, mask, affine = resample_acquisition(data, mask, affine, target, "gaussian")
        acquisition = {"spacing_mm": target.tolist(), "profile": "gaussian"}
    else:
        fwhm = rng.uniform(3, 5.5)
        data = ndimage.gaussian_filter(data, fwhm / (2.355 * nib.affines.voxel_sizes(affine)))
        acquisition = {"fwhm_mm": fwhm}

    data = np.where(mask, np.maximum(data, 0), 0).astype(np.float32)
    result = nib.Nifti1Image(data, affine)
    result.set_qform(affine, code=1)
    result.set_sform(affine, code=1)
    nib.save(result, output_path)
    return {
        "subject": subject,
        "age": age,
        "path": output_path,
        "params": json.dumps(
            {
                "family": family,
                "angles_deg": angles.tolist(),
                "shift_mm": shift.tolist(),
                "scale": scale,
                "gamma": gamma,
                "bias": bias.tolist(),
                "noise_sigma": noise_sigma,
                "ghost_axis": ghost_axis,
                "ghost_shift_mm": ghost_shift[ghost_axis],
                "ghost_weight": ghost_weight,
                "strip_erosion_mm": erosion,
                **acquisition,
            },
            separators=(",", ":"),
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_manifest", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    inputs = (
        pd.read_csv(args.input_manifest, sep="\t").sort_values("subject").reset_index(drop=True)
    )
    assert list(inputs.columns) == ["subject", "age", "path"]
    assert len(inputs) >= 128
    selected = np.sort(np.random.default_rng(SEED).choice(len(inputs), 128, replace=False))
    inputs = inputs.iloc[selected]
    (args.output_dir / "images").mkdir(parents=True, exist_ok=True)
    jobs = [
        (
            row.subject,
            int(row.age),
            row.path,
            str((args.output_dir / "images" / f"{row.subject}.nii.gz").resolve()),
        )
        for row in inputs.itertuples(index=False)
    ]
    with ProcessPoolExecutor(max_workers=32) as pool:
        rows = list(pool.map(augment_subject, jobs))
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "eval.tsv", sep="\t", index=False)
    with zipfile.ZipFile(args.output_dir / "Task_3_DLBS.zip", "w") as zf:
        for row in frame.itertuples(index=False):
            root = "Task_3_DLBS"
            zf.write(row.path, f"{root}/preprocessed/{row.subject}/ses-01/t1w.nii.gz")
            zf.writestr(f"{root}/labels/{row.subject}/ses-01/labels.txt", str(row.age))
            zf.writestr(f"{root}/params/{row.subject}/ses-01/params.json", row.params)
    print(f"wrote {len(rows)} DLBS evaluation images to {args.output_dir}")
