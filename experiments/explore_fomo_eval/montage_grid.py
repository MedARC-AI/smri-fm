"""Per-task grids: one view per task, subjects down rows, slices across columns."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

ROOT = Path(__file__).parents[2]
DATA_ROOT = ROOT / "data/fomo_eval"

VIEW_AXIS = {"axial": 2, "coronal": 1, "sagittal": 0}

# Per task: which view and which modality to show.
TASK_SPEC = {
    "Task_1": {"view": "axial", "base": "dwi_b1000"},
    "Task_2": {"view": "axial", "base": "flair"},
    "Task_3": {"view": "axial", "base": "t1w"},
    "Task_4": {"view": "axial", "base": "t2w"},
    "Task_5": {"view": "coronal", "base": "t1"},
}


def load_ras(path: Path) -> np.ndarray:
    return np.asarray(nib.as_closest_canonical(nib.load(path)).dataobj, dtype=np.float32)


def robust_norm(x: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(x, [1, 99])
    return np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)


def sessions(task: str) -> list[Path]:
    task_dir = DATA_ROOT / task / "preprocessed"
    return sorted(ses for sub in sorted(task_dir.iterdir()) for ses in sorted(sub.iterdir()))


def labels_dir(ses: Path) -> Path:
    return Path(str(ses).replace("/preprocessed/", "/labels/"))


def label_text(ses: Path) -> str:
    for name in ("label.txt", "labels.txt"):
        if (labels_dir(ses) / name).exists():
            return (labels_dir(ses) / name).read_text().strip()
    return "NA"


def load_seg(ses: Path) -> np.ndarray | None:
    seg_path = labels_dir(ses) / "seg.nii.gz"
    return load_ras(seg_path) if seg_path.exists() else None


def foreground_bbox(vol: np.ndarray, axis: int, margin: float = 0.06) -> tuple[int, int, int, int]:
    """In-plane (row0, row1, col0, col1) bounding box of the foreground, with a margin."""
    thr = 0.1 * np.percentile(vol[vol > 0], 99)
    plane = (vol > thr).any(axis=axis)
    rows = np.where(plane.any(axis=1))[0]
    cols = np.where(plane.any(axis=0))[0]
    if len(rows) == 0:
        return 0, plane.shape[0] - 1, 0, plane.shape[1] - 1
    pad_r = round(margin * plane.shape[0])
    pad_c = round(margin * plane.shape[1])
    r0 = max(rows[0] - pad_r, 0)
    r1 = min(rows[-1] + pad_r, plane.shape[0] - 1)
    c0 = max(cols[0] - pad_c, 0)
    c1 = min(cols[-1] + pad_c, plane.shape[1] - 1)
    return r0, r1, c0, c1


def crop_pad_square(img: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Crop to bbox then zero-pad to a centered square (keeps trailing channel axis, if any)."""
    r0, r1, c0, c1 = bbox
    cropped = img[r0 : r1 + 1, c0 : c1 + 1]
    h, w = cropped.shape[:2]
    size = max(h, w)
    pad_h, pad_w = size - h, size - w
    pads = [(pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2)]
    pads += [(0, 0)] * (cropped.ndim - 2)
    return np.pad(cropped, pads)


def pick_slice_indices(base: np.ndarray, seg: np.ndarray | None, axis: int, n: int) -> list[int]:
    """Columns of slices: spanning the lesion when segmented, else spread through the brain."""
    other = tuple(a for a in range(3) if a != axis)
    dim = base.shape[axis]
    if seg is not None and seg.sum() > 0:
        per_slice = (seg > 0).sum(axis=other)
        occupied = np.where(per_slice > 0)[0]
        lo, hi = int(occupied[0]), int(occupied[-1])
        # Widen a thin lesion so we get n distinct slices with context, not duplicates.
        deficit = n - (hi - lo + 1)
        if deficit > 0:
            lo, hi = lo - deficit // 2, hi + (deficit - deficit // 2)
            lo, hi = lo - min(0, dim - 1 - hi), hi - min(0, lo)
    else:
        thr = 0.1 * np.percentile(base[base > 0], 99)
        per_slice = (base > thr).sum(axis=other)
        occupied = np.where(per_slice > 0.02 * per_slice.max())[0]
        lo, hi = np.percentile(occupied, [15, 85]).astype(int)
    lo, hi = max(lo, 0), min(hi, dim - 1)
    return [int(round(i)) for i in np.linspace(lo, hi, n)]


def grid(task: str, n_subjects: int, n_slices: int, batch_id: int, out: Path) -> None:
    spec = TASK_SPEC[task]
    axis = VIEW_AXIS[spec["view"]]
    all_ses = sessions(task)
    ses_list = all_ses[batch_id * n_subjects : (batch_id + 1) * n_subjects]
    if not ses_list:
        n_batches = -(-len(all_ses) // n_subjects)
        print(f"skipped {task}: batch {batch_id} past end ({n_batches} batches available)")
        return

    fig, axes = plt.subplots(
        len(ses_list), n_slices, figsize=(1.7 * n_slices, 1.8 * len(ses_list)), squeeze=False
    )
    for r, ses in enumerate(ses_list):
        base_vol = load_ras(ses / f"{spec['base']}.nii.gz")
        seg = load_seg(ses)
        bbox = foreground_bbox(base_vol, axis)
        indices = pick_slice_indices(base_vol, seg, axis, n_slices)
        for c, idx in enumerate(indices):
            ax = axes[r][c]
            ax.set_xticks([])
            ax.set_yticks([])
            cell = crop_pad_square(robust_norm(np.take(base_vol, idx, axis=axis)), bbox)
            ax.imshow(np.rot90(cell), cmap="gray")
            if seg is not None:
                seg_slice = crop_pad_square(np.take(seg, idx, axis=axis), bbox)
                if seg_slice.any():
                    ax.contour(np.rot90(seg_slice), levels=[0.5], colors="lime", linewidths=0.7)
            # Per-cell index: each row centers on its own subject, so slices differ across rows.
            ax.text(
                0.05,
                0.95,
                str(idx),
                transform=ax.transAxes,
                fontsize=6,
                color="yellow",
                va="top",
                ha="left",
            )
            if c == 0:
                ax.set_ylabel(f"{ses.parent.name}\ny={label_text(ses)}", fontsize=8)

    fig.suptitle(
        f"{task} batch {batch_id}  ({spec['view']}, {spec['base']}, seg=lime)", fontsize=12
    )
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=list(TASK_SPEC))
    parser.add_argument("--n-subjects", type=int, default=5)
    parser.add_argument("--n-slices", type=int, default=6)
    parser.add_argument("--batch-id", type=int, nargs="+", default=[0])
    parser.add_argument("--out-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for task in args.tasks:
        for batch_id in args.batch_id:
            grid(
                task,
                args.n_subjects,
                args.n_slices,
                batch_id,
                args.out_dir / f"{task}_grid_b{batch_id}.png",
            )
