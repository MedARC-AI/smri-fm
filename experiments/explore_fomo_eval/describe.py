"""Print descriptive stats for the FOMO26 eval tasks: geometry, intensities, labels, segs."""

import argparse
import collections
from pathlib import Path

import nibabel as nib
import numpy as np

TASKS = [f"Task_{i}" for i in range(1, 6)]

ROOT = Path(__file__).parents[2]


def sessions(task_dir: Path) -> list[Path]:
    return sorted(ses for sub in sorted(task_dir.iterdir()) for ses in sorted(sub.iterdir()))


def modality(path: Path) -> str:
    return path.name.replace(".nii.gz", "")


def images_by_modality(task_dir: Path) -> dict[str, list[Path]]:
    by_mod = collections.defaultdict(list)
    for path in sorted(task_dir.rglob("*.nii.gz")):
        by_mod[modality(path)].append(path)
    return dict(by_mod)


def top_counts(values: list, k: int = 3) -> str:
    counter = collections.Counter(values)
    parts = [f"{value} x{count}" for value, count in counter.most_common(k)]
    suffix = f" (+{len(counter) - k} more)" if len(counter) > k else ""
    return ", ".join(parts) + suffix


def describe_layout(root: Path) -> None:
    print("=" * 100)
    print("LAYOUT")
    print("=" * 100)
    for task in TASKS:
        preproc = root / task / "preprocessed"
        ses_list = sessions(preproc)
        n_subjects = len(list(preproc.iterdir()))
        per_subject = collections.Counter(len(list(sub.iterdir())) for sub in preproc.iterdir())
        combos = collections.Counter(
            tuple(sorted(modality(f) for f in ses.glob("*.nii.gz"))) for ses in ses_list
        )
        print(f"\n{task}: {n_subjects} subjects, {len(ses_list)} sessions")
        print(f"  sessions per subject: {dict(per_subject)}")
        for combo, count in combos.most_common():
            print(f"  {count:4d}x {list(combo)}")


def describe_geometry(root: Path) -> None:
    print("\n" + "=" * 100)
    print("GEOMETRY (headers only)")
    print("=" * 100)
    for task in TASKS:
        print(f"\n{task}")
        for mod, paths in images_by_modality(root / task / "preprocessed").items():
            shapes, zooms, dtypes, orients = [], [], [], []
            for path in paths:
                img = nib.load(path)
                shapes.append(tuple(int(s) for s in img.shape))
                zooms.append(tuple(round(float(z), 3) for z in img.header.get_zooms()[:3]))
                dtypes.append(str(img.get_data_dtype()))
                orients.append("".join(nib.aff2axcodes(img.affine)))
            fov = [
                tuple(round(s * z, 1) for s, z in zip(shape, zoom))
                for shape, zoom in zip(shapes, zooms)
            ]
            print(f"  {mod:11s} n={len(paths)}")
            print(f"    shape   {top_counts(shapes)}")
            print(f"    spacing {top_counts(zooms)}")
            print(f"    fov_mm  {top_counts(fov)}")
            print(f"    dtype   {top_counts(dtypes)}   orient {top_counts(orients)}")


def describe_intensities(root: Path, n_sample: int) -> None:
    print("\n" + "=" * 100)
    print(f"INTENSITIES (first {n_sample} volumes per modality; ranges are across volumes)")
    print("=" * 100)
    header = f"  {'modality':22s} {'min':>16s} {'max':>20s} {'mean':>18s} {'p99':>20s}"
    print(header)
    for task in TASKS:
        for mod, paths in images_by_modality(root / task / "preprocessed").items():
            stats = collections.defaultdict(list)
            for path in paths[:n_sample]:
                vol = np.asarray(nib.load(path).dataobj, dtype=np.float32)
                stats["min"].append(float(vol.min()))
                stats["max"].append(float(vol.max()))
                stats["mean"].append(float(vol.mean()))
                stats["p99"].append(float(np.percentile(vol, 99)))
            spans = {k: f"[{min(v):.2f}, {max(v):.2f}]" for k, v in stats.items()}
            print(
                f"  {task + '/' + mod:22s} {spans['min']:>16s} {spans['max']:>20s} "
                f"{spans['mean']:>18s} {spans['p99']:>20s}"
            )


def describe_segmentations(root: Path) -> None:
    print("\n" + "=" * 100)
    print("SEGMENTATIONS")
    print("=" * 100)
    for task in TASKS:
        seg_paths = sorted((root / task / "labels").rglob("seg.nii.gz"))
        if not seg_paths:
            continue
        values, fractions, voxels, dtypes, grid_match = set(), [], [], [], []
        for seg_path in seg_paths:
            seg = nib.load(seg_path)
            arr = np.asarray(seg.dataobj)
            values |= {float(v) for v in np.unique(arr)}
            n_fg = int((arr > 0).sum())
            voxels.append(n_fg)
            fractions.append(n_fg / arr.size)
            dtypes.append(str(seg.get_data_dtype()))
            image_dir = Path(str(seg_path.parent).replace("/labels/", "/preprocessed/"))
            image = nib.load(sorted(image_dir.glob("*.nii.gz"))[0])
            grid_match.append(image.shape == seg.shape)
        n_sessions = len(sessions(root / task / "preprocessed"))
        frac, vox = np.array(fractions), np.array(voxels)
        print(f"\n{task}: {len(seg_paths)} segs for {n_sessions} sessions")
        print(f"  values         {sorted(values)}")
        print(f"  dtype          {top_counts(dtypes)}")
        print(f"  matches image grid: {sum(grid_match)}/{len(grid_match)}")
        print(
            f"  fg fraction    min={frac.min():.2e}  median={np.median(frac):.2e}  "
            f"max={frac.max():.2e}"
        )
        print(f"  fg voxels      min={vox.min()}  median={np.median(vox):.0f}  max={vox.max()}")


def describe_targets(root: Path) -> None:
    print("\n" + "=" * 100)
    print("TARGETS")
    print("=" * 100)
    for task in TASKS:
        label_paths = sorted(p for p in (root / task / "labels").rglob("label*.txt") if p.is_file())
        if not label_paths:
            continue
        raw = [p.read_text().strip() for p in label_paths]
        counts = collections.Counter(raw)
        print(f"\n{task}: {len(raw)} labels, {len(counts)} unique")
        if len(counts) <= 10:
            for value, count in sorted(counts.items()):
                print(f"  class {value}: {count:4d} ({count / len(raw):.1%})")
            majority = counts.most_common(1)[0][1] / len(raw)
            print(f"  majority-class accuracy: {majority:.3f}   (AUROC 0.5)")
        else:
            values = np.array([float(v) for v in raw])
            print(
                f"  min={values.min():.1f} max={values.max():.1f} "
                f"mean={values.mean():.1f} std={values.std():.1f} "
                f"median={np.median(values):.1f}"
            )
            edges = np.arange(np.floor(values.min() / 5) * 5, values.max() + 10, 5)
            hist, _ = np.histogram(values, bins=edges)
            for count, edge in zip(hist, edges):
                bar = "#" * int(count / max(hist.max() / 40, 1))
                print(f"    {int(edge):3d}-{int(edge) + 4:3d} {bar:40s} {count}")
            print(f"  predict-mean MAE:   {np.abs(values - values.mean()).mean():.2f}")
            print(f"  predict-median MAE: {np.abs(values - np.median(values)).mean():.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=(ROOT / "data/fomo_eval"))
    parser.add_argument("--n-intensity-sample", type=int, default=12)
    parser.add_argument(
        "--sections",
        nargs="+",
        default=["layout", "geometry", "intensities", "segmentations", "targets"],
    )
    args = parser.parse_args()

    if "layout" in args.sections:
        describe_layout(args.data_root)
    if "geometry" in args.sections:
        describe_geometry(args.data_root)
    if "intensities" in args.sections:
        describe_intensities(args.data_root, args.n_intensity_sample)
    if "segmentations" in args.sections:
        describe_segmentations(args.data_root)
    if "targets" in args.sections:
        describe_targets(args.data_root)
