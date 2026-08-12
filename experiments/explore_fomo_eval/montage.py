"""Render slice montages of the FOMO26 eval tasks, one PNG per task."""

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


def load_ras(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    img = nib.as_closest_canonical(nib.load(path))
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    return np.asarray(img.dataobj, dtype=np.float32), zooms


def take_slice(vol: np.ndarray, view: str, frac: float) -> np.ndarray:
    """Slice an RAS (X=L-R, Y=P-A, Z=I-S) volume, rotated for radiological-ish display."""
    axis = VIEW_AXIS[view]
    idx = min(round(frac * vol.shape[axis]), vol.shape[axis] - 1)
    return np.rot90(np.take(vol, idx, axis=axis))


def slice_aspect(zooms: tuple[float, float, float], view: str) -> float:
    """Height/width mm-per-pixel ratio, so anisotropic volumes render undistorted."""
    remaining = [z for a, z in enumerate(zooms) if a != VIEW_AXIS[view]]
    return remaining[1] / remaining[0]


def robust_norm(sl: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(sl, [1, 99])
    return np.clip((sl - lo) / max(hi - lo, 1e-6), 0, 1)


def sessions(task: str, group: str) -> list[Path]:
    task_dir = DATA_ROOT / task / group
    return sorted(ses for sub in sorted(task_dir.iterdir()) for ses in sorted(sub.iterdir()))


def seg_slice(ses: Path, view: str, frac: float) -> np.ndarray | None:
    seg_path = Path(str(ses).replace("/preprocessed/", "/labels/")) / "seg.nii.gz"
    if not seg_path.exists():
        return None
    return take_slice(load_ras(seg_path)[0], view, frac)


def label_text(ses: Path) -> str:
    label_dir = Path(str(ses).replace("/preprocessed/", "/labels/"))
    for name in ("label.txt", "labels.txt"):
        if (label_dir / name).exists():
            return (label_dir / name).read_text().strip()
    return "NA"


def seg_peak_frac(ses: Path, view: str) -> float:
    """Slice fraction at the most foreground voxels."""
    seg_path = Path(str(ses).replace("/preprocessed/", "/labels/")) / "seg.nii.gz"
    if not seg_path.exists():
        return 0.5
    seg = load_ras(seg_path)[0]
    axis = VIEW_AXIS[view]
    other_axes = tuple(a for a in range(3) if a != axis)
    per_slice_fg = (seg > 0).sum(axis=other_axes)
    return float(per_slice_fg.argmax()) / seg.shape[axis]


def montage(task: str, view: str, n_subjects: int, batch_id: int, out: Path) -> None:
    all_sessions = sessions(task, "preprocessed")
    ses_list = all_sessions[batch_id * n_subjects : (batch_id + 1) * n_subjects]
    if not ses_list:
        n_batches = -(-len(all_sessions) // n_subjects)
        print(f"skipped {task}: batch {batch_id} is past the end ({n_batches} batches available)")
        return
    modalities = sorted({f.name for ses in ses_list for f in ses.glob("*.nii.gz")})

    n_rows, n_cols = len(ses_list), len(modalities)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 2.8 * n_rows), squeeze=False)

    for r, ses in enumerate(ses_list):
        frac = seg_peak_frac(ses, view)
        seg = seg_slice(ses, view, frac)
        for c, mod in enumerate(modalities):
            ax = axes[r][c]
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(f"{ses.parent.name}\ny={label_text(ses)}", fontsize=8)
            if r == 0:
                ax.set_title(mod.replace(".nii.gz", ""), fontsize=10)
            path = ses / mod
            if not path.exists():
                ax.set_frame_on(False)
                continue
            vol, zooms = load_ras(path)
            aspect = slice_aspect(zooms, view)
            ax.imshow(robust_norm(take_slice(vol, view, frac)), cmap="gray", aspect=aspect)
            if seg is not None:
                ax.imshow(
                    np.ma.masked_where(seg == 0, seg),
                    cmap="autumn",
                    alpha=0.3,
                    vmin=0,
                    vmax=2,
                    aspect=aspect,
                )

    fig.suptitle(
        f"{task} batch {batch_id}  ({view}, 1-99 pct window, seg overlay in red/yellow)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=90, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=[f"Task_{i}" for i in range(1, 6)])
    parser.add_argument("--view", choices=list(VIEW_AXIS), default="axial")
    parser.add_argument("--n-subjects", type=int, default=5)
    parser.add_argument("--batch-id", type=int, nargs="+", default=[0])
    parser.add_argument("--out-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for task in args.tasks:
        for batch_id in args.batch_id:
            out = args.out_dir / f"{task}_{args.view}_b{batch_id}.png"
            montage(task, args.view, args.n_subjects, batch_id, out)
