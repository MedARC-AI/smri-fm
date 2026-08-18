"""A slice montage per class, to look for the cortical features SynthSeg's labels smooth away.

One row per subject, `--slices` columns centred on the widest slice and `--step-mm` apart,
cropped to the SynthSeg brain mask and zeroed outside it so the noise bands in some exports do
not dominate the intensity window.

The scans are acquired coronal, so `--plane coronal` is the acquired plane and the only one with
sub-millimetre spacing in both in-plane axes; axial and sagittal are reformats across the slice
direction. `--native` slices the original nifti instead of SynthSeg's 1mm resampling, carrying the
mask over by affine, which is worth 2x in plane on the Cigna subjects.

    uv run python experiments/explore_fomo_task5/montage.py --label 1 --plane axial
    uv run python experiments/explore_fomo_task5/montage.py --label 1 --plane coronal --native
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation, zoom

from segment import TASK_DIR, scanner, subjects

# slice axis, and whether the in-plane pair needs transposing to put rows on the second of them
PLANES = {"axial": 2, "coronal": 1, "sagittal": 0}


def on_grid(seg_img: nib.Nifti1Image, native_img: nib.Nifti1Image) -> np.ndarray:
    """SynthSeg's 1mm label map resampled onto the native grid by nearest neighbour.

    Both grids are canonical RAS with diagonal affines, so the mapping factors into one index
    lookup per axis rather than a general resample.
    """
    seg = np.asarray(seg_img.dataobj)
    source, target = np.asarray(seg_img.affine), np.asarray(native_img.affine)
    index = []
    for axis in range(3):
        world = target[axis, axis] * np.arange(native_img.shape[axis]) + target[axis, 3]
        position = (world - source[axis, 3]) / source[axis, axis]
        index.append(np.clip(np.rint(position).astype(int), 0, seg.shape[axis] - 1))
    return seg[index[0][:, None, None], index[1][None, :, None], index[2][None, None, :]]


def panels(
    t1: np.ndarray,
    brain: np.ndarray,
    plane: str,
    n: int,
    size: int,
    step: float,
    spacing: float,
) -> np.ndarray:
    """`n` slices `step` mm apart centred on the widest one, masked, cropped, resized."""
    mask = binary_dilation(brain, iterations=3)
    data = np.where(mask, t1, 0.0)
    low, high = np.percentile(t1[brain], (1, 99.5))
    data = np.clip((data - low) / (high - low), 0, 1)

    axis = PLANES[plane]
    box = [slice(int(w.min()), int(w.max()) + 1) for w in map(np.flatnonzero, extents(mask))]
    live = np.flatnonzero(extents(brain)[axis])
    # centred on the widest slice: the ends of the brain are slivers with no cortical detail
    area = brain.sum(axis=tuple(a for a in range(3) if a != axis))
    offsets = (np.arange(n) - (n - 1) / 2) * max(1, round(step / spacing))
    span = np.clip(int(np.argmax(area)) + offsets, live[0], live[-1])

    out = []
    for k in np.rint(span).astype(int):
        cut = [box[0], box[1], box[2]]
        cut[axis] = slice(k, k + 1)
        image = data[tuple(cut)].squeeze(axis).T[::-1]
        out.append(zoom(image, size / max(image.shape), order=1))
    height = max(p.shape[0] for p in out)
    width = max(p.shape[1] for p in out)
    return np.stack([pad(p, height, width) for p in out])


def extents(mask: np.ndarray) -> list[np.ndarray]:
    return [mask.any(axis=tuple(a for a in range(3) if a != axis)) for axis in range(3)]


def pad(image: np.ndarray, height: int, width: int) -> np.ndarray:
    top = (height - image.shape[0]) // 2
    left = (width - image.shape[1]) // 2
    out = np.zeros((height, width))
    out[top : top + image.shape[0], left : left + image.shape[1]] = image
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    here = Path(__file__).parent
    parser.add_argument("--out", type=Path, default=here / "output")
    parser.add_argument("--figures", type=Path, default=here / "figures")
    parser.add_argument("--label", type=int, default=1, help="1 for PMG, 0 for control")
    parser.add_argument("--plane", default="axial", choices=list(PLANES))
    parser.add_argument("--slices", type=int, default=8)
    parser.add_argument("--step-mm", type=float, default=8.0, help="spacing between slices")
    parser.add_argument("--size", type=int, default=192, help="pixels on a panel's long side")
    parser.add_argument("--native", action="store_true", help="slice the original grid")
    args = parser.parse_args()

    rows = [s for s, y in subjects() if y == args.label and (args.out / s / "seg.nii.gz").exists()]
    grid, names = [], []
    for sub in rows:
        seg_img = nib.load(args.out / sub / "seg.nii.gz")
        if args.native:
            path = TASK_DIR / f"preprocessed/{sub}/ses_01/t1.nii.gz"
            image = nib.as_closest_canonical(nib.load(path))
            brain = on_grid(seg_img, image) > 0
        else:
            image = nib.load(args.out / sub / "t1_1mm.nii.gz")
            brain = np.asarray(seg_img.dataobj) > 0
        t1 = np.asarray(image.dataobj, dtype=np.float32)
        spacing = float(image.header.get_zooms()[PLANES[args.plane]])
        grid.append(panels(t1, brain, args.plane, args.slices, args.size, args.step_mm, spacing))
        names.append(f"{sub}  {scanner(sub)}")
        print(f"{sub} {grid[-1].shape}", flush=True)

    height = max(g.shape[1] for g in grid)
    width = max(g.shape[2] for g in grid)
    montage = np.concatenate(
        [np.concatenate([pad(p, height, width) for p in g], axis=1)[None] for g in grid]
    )
    montage = montage.reshape(-1, montage.shape[-1])

    fig, ax = plt.subplots(figsize=(montage.shape[1] / 100, montage.shape[0] / 100))
    ax.imshow(montage, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_yticks(np.arange(len(rows)) * height + height / 2, names, fontsize=9)
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(
        f"task 5, {'PMG' if args.label else 'control'}, {args.plane}"
        f"{', native grid' if args.native else ', 1mm'}",
        fontsize=12,
    )
    fig.tight_layout()
    # jpeg, not png: these are 28 megapixel photographic greys and png costs 20MB of repo each
    name = f"{'pmg' if args.label else 'control'}_{args.plane}{'_native' if args.native else ''}"
    fig.savefig(args.figures / f"{name}.jpg", dpi=100, pil_kwargs={"quality": 92})
    plt.close(fig)


if __name__ == "__main__":
    main()
