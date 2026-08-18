"""Render the cortical surface of each task-5 subject from its SynthSeg cortex label.

No mesh library in the environment, and none is needed for an orthographic view: cast rays along
one axis, take the first cortex voxel, and shade the resulting depth map. Surface normals come
from the depth gradient (Lambertian), and a cavity term darkens whatever sits deeper than its
neighbourhood, which is what makes sulci read. Nothing is smoothed or decimated, so gyral detail
survives at the resolution SynthSeg labels it.

    uv run python experiments/explore_fomo_task5/cortex.py    # -> figures/cortex_*.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter

from segment import subjects

CORTEX = {"L": 3, "R": 42}

# name -> (hemisphere, sign of the x axis the camera looks along)
VIEWS = {
    "L lateral": ("L", +1),
    "L medial": ("L", -1),
    "R lateral": ("R", -1),
    "R medial": ("R", +1),
}


def depth_map(mask: np.ndarray, sign: int) -> np.ndarray:
    """First surface voxel along x, as an image with z up and the anterior pole facing the camera.

    `sign` is the x direction the camera looks along: +1 views the head from its left.
    Rays that never hit the mask come back as nan.
    """
    ordered = mask if sign > 0 else mask[::-1]
    depth = np.where(ordered.any(axis=0), ordered.argmax(axis=0).astype(float), np.nan)
    depth = depth.T  # (y, z) -> rows z, columns y, drawn with origin="lower"
    return depth[:, ::-1] if sign > 0 else depth


def shade(depth: np.ndarray, light=(-1.0, -0.4, 0.5), ambient: float = 0.25) -> np.ndarray:
    """Lambertian on the depth gradient, times a cavity term. nan outside the silhouette."""
    hit = np.isfinite(depth)
    filled = np.where(hit, depth, np.nanmax(depth))

    d_row, d_col = np.gradient(filled)
    normal = np.stack([-np.ones_like(filled), d_col, d_row])
    normal /= np.linalg.norm(normal, axis=0)
    light = np.asarray(light) / np.linalg.norm(light)
    lambert = np.clip(np.einsum("i...,i->...", normal, light), 0.0, 1.0)

    cavity = np.clip(1.0 - 0.12 * (filled - gaussian_filter(filled, 4.0)), 0.2, 1.0)
    return np.where(hit, np.clip((ambient + (1 - ambient) * lambert) * cavity, 0, 1), np.nan)


def render(seg: np.ndarray, view: str, margin: int = 4) -> np.ndarray:
    hemi, sign = VIEWS[view]
    image = shade(depth_map(seg == CORTEX[hemi], sign))
    rows, cols = np.where(np.isfinite(image))
    box = (
        slice(max(rows.min() - margin, 0), rows.max() + margin),
        slice(max(cols.min() - margin, 0), cols.max() + margin),
    )
    return image[box]


def draw(ax, image: np.ndarray, title: str) -> None:
    ax.imshow(image, cmap="gray", vmin=0, vmax=1, origin="lower", interpolation="nearest")
    ax.set_title(title, fontsize=8)
    ax.set_axis_off()


def main() -> None:
    parser = argparse.ArgumentParser()
    here = Path(__file__).parent
    parser.add_argument("--out", type=Path, default=here / "output")
    parser.add_argument("--figures", type=Path, default=here / "figures")
    args = parser.parse_args()
    args.figures.mkdir(parents=True, exist_ok=True)

    rows = [(sub, label) for sub, label in subjects() if (args.out / sub / "seg.nii.gz").exists()]
    segs = {sub: np.asarray(nib.load(args.out / sub / "seg.nii.gz").dataobj) for sub, _ in rows}

    for view in ("L lateral", "R lateral"):
        fig, axes = plt.subplots(6, 8, figsize=(20, 18))
        for ax, (sub, label) in zip(axes.flat, rows):
            draw(ax, render(segs[sub], view), f"{sub}  {'PMG' if label else 'control'}")
        for ax in axes.flat[len(rows) :]:
            ax.set_axis_off()
        fig.suptitle(f"task 5 cortex, {view}")
        fig.tight_layout()
        fig.savefig(args.figures / f"cortex_{view.replace(' ', '_').lower()}.png", dpi=140)
        plt.close(fig)

    controls = [sub for sub, label in rows if not label][:4]
    cases = [sub for sub, label in rows if label][:4]
    fig, axes = plt.subplots(len(VIEWS), 8, figsize=(20, 11))
    for row, view in zip(axes, VIEWS):
        for ax, sub in zip(row, controls + cases):
            label = dict(rows)[sub]
            draw(ax, render(segs[sub], view), f"{sub} {'PMG' if label else 'ctrl'}  {view}")
    fig.suptitle("task 5 cortex, four views, four controls then four cases")
    fig.tight_layout()
    fig.savefig(args.figures / "cortex_views.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
