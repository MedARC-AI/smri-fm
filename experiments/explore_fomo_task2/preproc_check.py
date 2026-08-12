"""Does the fomo_tune preprocessing keep the task-2 tumours?

Two failure modes, both fatal to a patch-based segmentation and neither visible in the score:
the 208x240x208 fit crops a peripheral tumour off the edge of the box, or the mean-intensity
brain mask excludes it -- outside the mask the transform writes zeros, so the tumour is erased.

Renders every subject through the real transform (dwi, flair, flair + tumour + mask contour)
and reports the two fractions per subject.
"""

import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from matplotlib.colors import ListedColormap

from fomo_tune.backbone import fit_to_shape, rescale

ROOT = Path(__file__).parents[2]
TASK_DIR = ROOT / "data/fomo_eval/Task_2"
IMG_SIZE = (208, 240, 208)
MODALITIES = ("dwi_b1000", "flair")
N_SLICES = 10


def preprocess(img: nib.Nifti1Image) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The SmriMaeTransform geometry and mask, as volume, brain mask, and output affine."""
    img = nib.as_closest_canonical(img)
    data = torch.from_numpy(np.ascontiguousarray(img.get_fdata(dtype=np.float32)))
    affine = np.asarray(img.affine)

    spacing = img.header.get_zooms()
    if max(abs(s - 1.0) for s in spacing) > 0.05:
        data, affine = rescale(data, affine, spacing)
    data, affine = fit_to_shape(data, affine, target_shape=IMG_SIZE)

    mask = data > data.mean()
    return data.numpy(), mask.numpy(), affine


def axis_lookup(affine_in: np.ndarray, affine_out: np.ndarray, shape_out: tuple[int, ...]):
    """Per-axis input index for each output index. Every step is axis-aligned, so the map is
    diagonal and the 3D nearest-neighbour resample factors into three 1D gathers."""
    transform = np.linalg.inv(affine_in) @ affine_out
    scale, offset = np.diag(transform)[:3], transform[:3, 3]
    assert np.allclose(transform[:3, :3], np.diag(scale), atol=1e-5), "map is not axis-aligned"
    return [np.round(scale[a] * np.arange(shape_out[a]) + offset[a]).astype(int) for a in range(3)]


def resample_seg(seg: np.ndarray, lookup: list[np.ndarray], shape_in: tuple[int, ...]):
    """Nearest-neighbour seg on the output grid, plus the input window the output can reach."""
    inside = [(idx >= 0) & (idx < dim) for idx, dim in zip(lookup, shape_in)]
    clipped = [np.clip(idx, 0, dim - 1) for idx, dim in zip(lookup, shape_in)]
    out = seg[np.ix_(*clipped)]
    out[~inside[0]] = 0
    out[:, ~inside[1]] = 0
    out[:, :, ~inside[2]] = 0
    window = [(idx[ok].min(), idx[ok].max()) for idx, ok in zip(lookup, inside)]
    return out, window


def retained_fraction(seg: np.ndarray, window: list[tuple[int, int]]) -> float:
    """Fraction of native tumour voxels inside the window the 208x240x208 box can see."""
    fg = np.argwhere(seg > 0)
    kept = np.ones(len(fg), dtype=bool)
    for a, (lo, hi) in enumerate(window):
        kept &= (fg[:, a] >= lo) & (fg[:, a] <= hi)
    return float(kept.mean())


def slice_columns(seg_out: np.ndarray, stride: int) -> np.ndarray:
    """Axial slices one acquired slice apart, centred on the largest tumour cross-section."""
    per_slice = (seg_out > 0).sum(axis=(0, 1))
    centre = int(per_slice.argmax())
    columns = centre + stride * (np.arange(N_SLICES) - N_SLICES // 2)
    shift = max(0, -columns[0]) - max(0, columns[-1] - (IMG_SIZE[2] - 1))
    return columns + shift


def window_levels(volume: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(volume[mask], [1, 99])
    return float(lo), float(hi)


def show(ax, plane: np.ndarray, levels: tuple[float, float]) -> None:
    ax.imshow(np.rot90(plane), cmap="gray", vmin=levels[0], vmax=levels[1])
    ax.set_xticks([])
    ax.set_yticks([])


def main() -> None:
    subjects = sorted(p.name for p in (TASK_DIR / "preprocessed").iterdir())
    fig, axes = plt.subplots(
        3 * len(subjects),
        N_SLICES,
        figsize=(1.2 * N_SLICES, 1.4 * 3 * len(subjects)),
        squeeze=False,
    )
    rows = []

    for s, subject in enumerate(subjects):
        session = TASK_DIR / f"preprocessed/{subject}/ses-01"
        seg_img = nib.as_closest_canonical(
            nib.load(TASK_DIR / f"labels/{subject}/ses-01/seg.nii.gz")
        )
        seg = np.asarray(seg_img.dataobj, dtype=np.float32).round()

        volumes, masks, levels = {}, {}, {}
        for modality in MODALITIES:
            img = nib.as_closest_canonical(nib.load(session / f"{modality}.nii.gz"))
            assert img.shape == seg_img.shape, f"{subject}: seg grid differs from {modality}"
            volumes[modality], masks[modality], affine_out = preprocess(img)
            levels[modality] = window_levels(volumes[modality], masks[modality])
            if modality == "flair":
                # the seg's own affine differs from the image's by up to 0.03mm; the image's is
                # what the transform used, and the two grids match
                lookup = axis_lookup(img.affine, affine_out, IMG_SIZE)
                stride = int(round(img.header.get_zooms()[2]))

        seg_out, window = resample_seg(seg, lookup, seg.shape)
        fg_out = seg_out > 0
        record = {
            "subject": subject,
            "fg_voxels": int((seg > 0).sum()),
            "retained": retained_fraction(seg, window),
            "fg_out": int(fg_out.sum()),
            **{f"in_mask_{m}": float(masks[m][fg_out].mean()) for m in MODALITIES},
        }
        rows.append(record)
        print(record, flush=True)

        for c, k in enumerate(slice_columns(seg_out, stride)):
            show(axes[3 * s][c], volumes["dwi_b1000"][:, :, k], levels["dwi_b1000"])
            show(axes[3 * s + 1][c], volumes["flair"][:, :, k], levels["flair"])
            ax = axes[3 * s + 2][c]
            show(ax, volumes["flair"][:, :, k], levels["flair"])
            ax.contour(
                np.rot90(masks["flair"][:, :, k]), levels=[0.5], colors="cyan", linewidths=0.4
            )
            plane = fg_out[:, :, k]
            if plane.any():
                ax.imshow(
                    np.rot90(np.ma.masked_where(~plane, plane)),
                    cmap=ListedColormap(["red"]),
                    alpha=0.5,
                )
                ax.contour(np.rot90(plane), levels=[0.5], colors="red", linewidths=0.6)
            axes[3 * s][c].set_title(f"z={k}", fontsize=5)

        axes[3 * s][0].set_ylabel(f"{subject}\ndwi  n={record['fg_voxels']}", fontsize=6)
        axes[3 * s + 1][0].set_ylabel(f"flair\nkept={record['retained']:.2f}", fontsize=6)
        axes[3 * s + 2][0].set_ylabel(
            f"seg+mask\nin_mask={record['in_mask_flair']:.2f}", fontsize=6
        )

    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    fig.suptitle(f"Task 2 through the fomo_tune transform (git {sha.stdout.strip()})", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.996))
    out_dir = Path(__file__).parent / "figures"
    out_dir.mkdir(exist_ok=True)
    fig.savefig(out_dir / "preproc_check.png", dpi=100, bbox_inches="tight")

    header = ["subject", "fg_voxels", "retained", "fg_out", *[f"in_mask_{m}" for m in MODALITIES]]
    lines = ["\t".join(header)]
    lines += [
        "\t".join(f"{r[k]:.3f}" if isinstance(r[k], float) else str(r[k]) for k in header)
        for r in rows
    ]
    (Path(__file__).parent / "preproc_check.tsv").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
