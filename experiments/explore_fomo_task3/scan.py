"""Per-subject scalars and one axial slice for each of the 494 task 3 t1w volumes.

Everything here is model-free: it is what a reader could measure off the images without a
backbone, and it is what the error analysis in `explore.py` correlates the predictions against.
The images arrive skull-stripped on a common 176x256x256 1mm RAS grid, so `data > 0` is a brain
mask and voxel counts are already mL.

CPU only, ~2 minutes on 32 workers.
"""

from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage, stats

ROOT = Path(__file__).parents[2]
TASK_DIR = ROOT / "data/fomo_eval/Task_3"
OUT_DIR = Path(__file__).parent
CROP_MM = (180, 216)
Z_FRACTION = 0.55
HIST_BINS = np.linspace(0, 2.5, 33)


def measure(subject: str) -> tuple[dict, np.ndarray, np.ndarray]:
    session = TASK_DIR / f"preprocessed/{subject}/ses-01"
    path = session / "t1w.nii.gz"
    img = nib.load(path)
    assert nib.aff2axcodes(img.affine) == ("R", "A", "S"), subject
    data = np.asarray(img.dataobj, dtype=np.float32)

    brain = data > 0
    inside = data[brain]
    median = np.median(inside)
    relative = inside / median

    box = [(idx.min(), idx.max() + 1) for idx in np.nonzero(brain)]
    centroid = ndimage.center_of_mass(brain)
    # gradient magnitude inside the brain: a protocol signature, low for smooth or thick-slice data
    gradient = np.gradient(data / median)
    sharpness = np.sqrt(sum(g[brain] ** 2 for g in gradient)).mean()

    record = {
        "subject": subject,
        "age": int((TASK_DIR / f"labels/{subject}/ses-01/labels.txt").read_text().strip()),
        "brain_ml": float(brain.sum() / 1000),
        "mask_ml": float((data > data.mean()).sum() / 1000),
        "bbox_x": float(box[0][1] - box[0][0]),
        "bbox_y": float(box[1][1] - box[1][0]),
        "bbox_z": float(box[2][1] - box[2][0]),
        "centroid_x": float(centroid[0]),
        "centroid_y": float(centroid[1]),
        "centroid_z": float(centroid[2]),
        # fill of the bounding box: falls as sulci widen, so an atrophy proxy independent of size
        "box_fill": float(brain.sum() / np.prod([hi - lo for lo, hi in box])),
        "median": float(median),
        "p05": float(np.percentile(relative, 5)),
        "p25": float(np.percentile(relative, 25)),
        "p75": float(np.percentile(relative, 75)),
        "p95": float(np.percentile(relative, 95)),
        "skew": float(stats.skew(relative)),
        "kurtosis": float(stats.kurtosis(relative)),
        # dark voxels inside the brain mask: csf, so ventricles and widened sulci
        "dark_frac": float((relative < 0.55).mean()),
        "sharpness": float(sharpness),
        "bytes": path.stat().st_size,
    }

    histogram, _ = np.histogram(relative, bins=HIST_BINS, density=True)

    z = int(box[2][0] + Z_FRACTION * (box[2][1] - box[2][0]))
    plane = np.zeros(CROP_MM, dtype=np.float32)
    starts = [int(centroid[axis]) - CROP_MM[axis] // 2 for axis in (0, 1)]
    source = [
        slice(max(0, s), min(dim, s + width))
        for s, width, dim in zip(starts, CROP_MM, data.shape[:2])
    ]
    target = [slice(src.start - s, src.stop - s) for src, s in zip(source, starts)]
    plane[target[0], target[1]] = data[source[0], source[1], z] / np.percentile(inside, 99)
    tile = np.rot90(plane)[::2, ::2]
    return record, histogram.astype(np.float32), (255 * tile.clip(0, 1)).astype(np.uint8)


def main() -> None:
    subjects = sorted(p.name for p in (TASK_DIR / "preprocessed").iterdir())
    with Pool(32) as pool:
        results = pool.map(measure, subjects)

    records = [r[0] for r in results]
    np.savez(
        OUT_DIR / "scan.npz",
        subjects=np.array(subjects),
        histogram=np.stack([r[1] for r in results]),
        tile=np.stack([r[2] for r in results]),
    )

    header = list(records[0])
    lines = ["\t".join(header)]
    for record in records:
        lines.append(
            "\t".join(
                f"{record[k]:.4f}" if isinstance(record[k], float) else str(record[k])
                for k in header
            )
        )
    (OUT_DIR / "scan.tsv").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:5]))


if __name__ == "__main__":
    main()
