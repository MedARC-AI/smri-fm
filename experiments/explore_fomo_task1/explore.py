"""What is task 1's AUROC 0.990 actually made of?

Pairs each subject's out-of-fold probability from the baseline log with what is in its images:
lesion size, location and conspicuity from the seg the positives carry, the acquisition geometry,
and scalars a model-free baseline could use. Renders every subject through the real transform so
the errors can be looked at. Findings are in README.md.
"""

import re
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from scipy import ndimage, stats
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fomo_tune.backbone import fit_to_shape, rescale

ROOT = Path(__file__).parents[2]
TASK_DIR = ROOT / "data/fomo_eval/Task_1"
LOG = ROOT / "experiments/fomo_tune_baseline/output/task1/log.txt"
OUT_DIR = Path(__file__).parent
IMG_SIZE = (208, 240, 208)
MODALITIES = ("dwi_b1000", "adc")
N_SLICES = 10
ZOOM_SLICES = 6
ZOOM_MM = 100


def preprocess(img: nib.Nifti1Image) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The SmriMaeTransform geometry, mask and normalization, plus the output affine."""
    img = nib.as_closest_canonical(img)
    data = torch.from_numpy(np.ascontiguousarray(img.get_fdata(dtype=np.float32)))
    affine = np.asarray(img.affine)

    spacing = img.header.get_zooms()
    if max(abs(s - 1.0) for s in spacing) > 0.05:
        data, affine = rescale(data, affine, spacing)
    data, affine = fit_to_shape(data, affine, target_shape=IMG_SIZE)

    mask = data > data.mean()
    brain = data[mask]
    normed = torch.where(mask, (data - brain.mean()) / brain.std(correction=0).clamp_min(1e-6), 0.0)
    return normed.numpy(), mask.numpy(), affine


def axis_lookup(affine_in: np.ndarray, affine_out: np.ndarray, shape_out: tuple[int, ...]):
    """Per-axis input index for each output index; every step of the transform is axis-aligned."""
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
    fg = np.argwhere(seg > 0)
    kept = np.ones(len(fg), dtype=bool)
    for a, (lo, hi) in enumerate(window):
        kept &= (fg[:, a] >= lo) & (fg[:, a] <= hi)
    return float(kept.mean())


def read_oof(log_path: Path) -> dict[str, tuple[int, float]]:
    text = log_path.read_text()
    folds = re.findall(r"(sub-\d+) y=(\d) p=([0-9.]+)", text)
    return {sub: (int(y), float(p)) for sub, y, p in folds}


def lesion_stats(seg: np.ndarray, zooms: tuple[float, ...]) -> dict:
    """Size and fragmentation in native voxels, where the seg was actually drawn."""
    fg = seg > 0
    components, n_components = ndimage.label(fg)
    sizes = np.bincount(components.ravel())[1:]
    return {
        "voxels": int(fg.sum()),
        "volume_ml": float(fg.sum() * np.prod(zooms) / 1000),
        "n_slices": int((fg.sum(axis=(0, 1)) > 0).sum()),
        "n_components": int(n_components),
        "largest_frac": float(sizes.max() / sizes.sum()),
    }


def location_stats(seg_out: np.ndarray, mask: np.ndarray) -> dict:
    """Lesion centroid in the brain-mask box, in mm from its centre and as an axis fraction.

    No atlas, so this is geometry only: +x is right, +y anterior, +z superior, and the frame is
    the transform's own mask bounding box rather than any anatomical landmark.
    """
    box = [(idx.min(), idx.max()) for idx in np.nonzero(mask)]
    centre = np.array([(lo + hi) / 2 for lo, hi in box])
    span = np.array([hi - lo for lo, hi in box], dtype=float)
    centroid = np.array(ndimage.center_of_mass(seg_out > 0))
    offset = centroid - centre
    fraction = (centroid - np.array([lo for lo, _ in box])) / span
    return {
        "x_mm": float(offset[0]),
        "y_mm": float(offset[1]),
        "z_mm": float(offset[2]),
        "x_frac": float(fraction[0]),
        "y_frac": float(fraction[1]),
        "z_frac": float(fraction[2]),
    }


def measure() -> list[dict]:
    oof = read_oof(LOG)
    subjects = sorted(p.name for p in (TASK_DIR / "preprocessed").iterdir())
    rows = []

    for subject in subjects:
        session = TASK_DIR / f"preprocessed/{subject}/ses-01"
        labels = TASK_DIR / f"labels/{subject}/ses-01"
        label, p = oof[subject]
        fourth = next(
            m.stem.split(".")[0] for m in session.iterdir() if m.name[:3] in ("swi", "t2s")
        )

        volumes, masks = {}, {}
        for modality in MODALITIES:
            img = nib.as_closest_canonical(nib.load(session / f"{modality}.nii.gz"))
            volumes[modality], masks[modality], affine_out = preprocess(img)
        shape, zooms = img.shape, img.header.get_zooms()

        mask = masks["dwi_b1000"]
        brain = volumes["dwi_b1000"][mask]
        record = {
            "subject": subject,
            "label": label,
            "p": p,
            "fourth": fourth,
            "shape": "x".join(str(s) for s in shape),
            "in_plane_mm": float(zooms[0]),
            "slice_mm": float(zooms[2]),
            "fov_x": float(shape[0] * zooms[0]),
            "fov_z": float(shape[2] * zooms[2]),
            # the images are skull-stripped, so the mask is brain and this is a volume in mL
            "brain_ml": float(mask.sum() / 1000),
            "dwi_p99": float(np.percentile(brain, 99)),
            "dwi_p999": float(np.percentile(brain, 99.9)),
            "dwi_above_z3": float((brain > 3).mean()),
            "dwi_skew": float(stats.skew(brain)),
        }

        if (labels / "seg.nii.gz").exists():
            seg_img = nib.as_closest_canonical(nib.load(labels / "seg.nii.gz"))
            assert seg_img.shape == img.shape, f"{subject}: seg grid differs from the images"
            seg = np.asarray(seg_img.dataobj, dtype=np.float32).round()
            lookup = axis_lookup(img.affine, affine_out, IMG_SIZE)
            seg_out, window = resample_seg(seg, lookup, seg.shape)
            fg_out = seg_out > 0

            record |= lesion_stats(seg, zooms)
            record |= location_stats(seg_out, masks["dwi_b1000"])
            record["retained"] = retained_fraction(seg, window)
            record["in_mask"] = float(masks["dwi_b1000"][fg_out].mean())
            # the transform's normalization makes a mean over the mask a z-score by construction
            for modality in MODALITIES:
                inside = masks[modality] & fg_out
                record[f"z_{modality}"] = float(volumes[modality][inside].mean())
            record["seg_out"] = fg_out
        else:
            record["seg_out"] = None

        record["volumes"] = volumes
        record["masks"] = masks
        rows.append(record)
        print(
            {k: v for k, v in record.items() if k not in ("volumes", "masks", "seg_out")},
            flush=True,
        )

    return rows


def permutation_p(y: np.ndarray, score: np.ndarray, seed: int = 0, n_perm: int = 20000) -> float:
    """One-sided: how often a label shuffle beats this AUROC. n=21 needs the exact-ish version."""
    rng = np.random.default_rng(seed)
    observed = roc_auc_score(y, score)
    null = np.array([roc_auc_score(rng.permutation(y), score) for _ in range(n_perm)])
    return float((null >= observed).mean())


def dumb_baselines(rows: list[dict]) -> None:
    """What the label can be read off without the backbone: geometry, brain size, DWI histogram."""
    y = np.array([r["label"] for r in rows])
    p = np.array([r["p"] for r in rows])

    print("\n-- single scalars, AUROC as-is / sign-flipped, with a permutation p")
    for key in ("brain_ml", "dwi_p99", "dwi_p999", "dwi_above_z3", "dwi_skew", "slice_mm", "fov_x"):
        v = np.array([r[key] for r in rows])
        auroc = roc_auc_score(y, v)
        best = v if auroc >= 0.5 else -v
        print(
            f"{key:14s} auroc={max(auroc, 1 - auroc):.3f} sign={'+' if auroc >= 0.5 else '-'} "
            f"perm_p={permutation_p(y, best):.4f}  spearman(model p)={spearmanr(p, v).statistic:+.3f}",
            flush=True,
        )

    print("\n-- leave-one-out logistic heads on non-backbone features")
    feature_sets = {
        "geometry": ["in_plane_mm", "slice_mm", "fov_x", "fov_z"],
        "brain_ml": ["brain_ml"],
        "dwi_hist": ["dwi_p99", "dwi_p999", "dwi_above_z3", "dwi_skew"],
        "all": [
            "in_plane_mm",
            "slice_mm",
            "fov_x",
            "fov_z",
            "brain_ml",
            "dwi_p99",
            "dwi_p999",
            "dwi_above_z3",
            "dwi_skew",
        ],
    }
    for name, keys in feature_sets.items():
        X = np.array([[r[k] for k in keys] for r in rows])
        oof = np.zeros(len(y))
        for held_out in range(len(y)):
            train = np.arange(len(y)) != held_out
            head = make_pipeline(
                StandardScaler(),
                LogisticRegressionCV(
                    Cs=10,
                    class_weight="balanced",
                    scoring="roc_auc",
                    max_iter=1000,
                    l1_ratios=(0,),
                    use_legacy_attributes=False,
                ),
            )
            head.fit(X[train], y[train])
            oof[held_out] = head.predict_proba(X[held_out : held_out + 1])[0, 1]
        print(f"{name:10s} auroc={roc_auc_score(y, oof):.3f}", flush=True)

    print("\n-- the model's own score, split by the 4th modality (a site proxy)")
    swi = np.array([r["fourth"] == "swi" for r in rows])
    for name, group in (("swi", swi), ("t2s", ~swi)):
        print(
            f"within {name}: n={group.sum()} pos={y[group].sum()} "
            f"auroc={roc_auc_score(y[group], p[group]):.3f} mean_p={p[group].mean():.4f}",
            flush=True,
        )


def show(ax, plane: np.ndarray, seg_plane: np.ndarray | None, mask_plane: np.ndarray) -> None:
    ax.imshow(np.rot90(plane), cmap="gray", vmin=-1.0, vmax=4.0)
    ax.contour(np.rot90(mask_plane), levels=[0.5], colors="cyan", linewidths=0.3)
    if seg_plane is not None and seg_plane.any():
        ax.contour(np.rot90(seg_plane), levels=[0.5], colors="red", linewidths=0.6)
    ax.set_xticks([])
    ax.set_yticks([])


def figure_subjects(rows: list[dict]) -> None:
    """Every subject, both modalities, axial slices spanning the brain box the model sees."""
    fig, axes = plt.subplots(
        2 * len(rows), N_SLICES, figsize=(1.2 * N_SLICES, 1.4 * 2 * len(rows)), squeeze=False
    )
    for s, record in enumerate(rows):
        mask = record["masks"]["dwi_b1000"]
        z = np.nonzero(mask.any(axis=(0, 1)))[0]
        columns = np.linspace(z.min(), z.max(), N_SLICES + 2)[1:-1].astype(int)
        for m, modality in enumerate(MODALITIES):
            for c, k in enumerate(columns):
                seg = record["seg_out"][:, :, k] if record["seg_out"] is not None else None
                show(axes[2 * s + m][c], record["volumes"][modality][:, :, k], seg, mask[:, :, k])
            axes[2 * s + m][0].set_ylabel(modality, fontsize=6)
        axes[2 * s][0].set_ylabel(
            f"{record['subject']}  y={record['label']}\np={record['p']:.3f}\ndwi", fontsize=6
        )
        for c, k in enumerate(columns):
            axes[2 * s][c].set_title(f"z={k}", fontsize=5)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "figures/subjects.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def figure_lesions(rows: list[dict]) -> None:
    """Positives only, cropped around the lesion, one 5-7mm acquired slice apart."""
    positives = [r for r in rows if r["seg_out"] is not None]
    fig, axes = plt.subplots(
        2 * len(positives),
        ZOOM_SLICES,
        figsize=(1.4 * ZOOM_SLICES, 1.5 * 2 * len(positives)),
        squeeze=False,
    )
    for s, record in enumerate(positives):
        fg = record["seg_out"]
        centre = np.array(ndimage.center_of_mass(fg)).round().astype(int)
        stride = int(round(record["slice_mm"]))
        columns = centre[2] + stride * (np.arange(ZOOM_SLICES) - ZOOM_SLICES // 2)
        columns = np.clip(columns, 0, IMG_SIZE[2] - 1)
        box = [
            slice(max(0, c - ZOOM_MM // 2), min(dim, c + ZOOM_MM // 2))
            for c, dim in zip(centre[:2], IMG_SIZE[:2])
        ]
        for m, modality in enumerate(MODALITIES):
            for c, k in enumerate(columns):
                ax = axes[2 * s + m][c]
                ax.imshow(
                    np.rot90(record["volumes"][modality][box[0], box[1], k]),
                    cmap="gray",
                    vmin=-1.0,
                    vmax=4.0,
                )
                plane = fg[box[0], box[1], k]
                if plane.any():
                    ax.contour(np.rot90(plane), levels=[0.5], colors="red", linewidths=0.6)
                ax.set_xticks([])
                ax.set_yticks([])
            axes[2 * s + m][0].set_ylabel(modality, fontsize=6)
        axes[2 * s][0].set_ylabel(
            f"{record['subject']} p={record['p']:.3f}\n{record['volume_ml']:.1f}mL\ndwi", fontsize=6
        )
        for c, k in enumerate(columns):
            axes[2 * s][c].set_title(f"z={k}", fontsize=5)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "figures/lesions.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


# colour is the label, marker is the 4th modality, everywhere except the location panel
GROUPS = (("swi", 0, "s"), ("swi", 1, "s"), ("t2s", 0, "o"), ("t2s", 1, "o"))


def group_scatter(ax, rows: list[dict], x_key: str, y_key: str) -> None:
    for fourth, label, marker in GROUPS:
        subset = [r for r in rows if r["label"] == label and r["fourth"] == fourth]
        if not subset:
            continue
        ax.scatter(
            [r[x_key] for r in subset],
            [r[y_key] for r in subset],
            color=f"C{label}",
            marker=marker,
            label=f"{fourth} y={label}",
        )
    for record in rows:
        ax.annotate(
            record["subject"].removeprefix("sub-"),
            (record[x_key], record[y_key]),
            fontsize=7,
            xytext=(6, 0),
            textcoords="offset points",
        )
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)


def figure_scores(rows: list[dict]) -> None:
    y = np.array([r["label"] for r in rows])
    p = np.array([r["p"] for r in rows])
    positives = [r for r in rows if r["seg_out"] is not None]
    for record in rows:
        record["label_x"] = record["label"] + (0.05 if record["fourth"] == "swi" else -0.05)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    ax = axes[0, 0]
    group_scatter(ax, rows, "label_x", "p")
    ax.axhline(p[y == 0].max(), ls=":", c="k", lw=0.8, label="max negative")
    ax.set_xticks([0, 1])
    ax.set_xlabel("label")
    ax.set_ylabel("out-of-fold p")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    group_scatter(ax, positives, "volume_ml", "p")
    ax.set_xscale("log")
    ax.set_xlabel("lesion volume (mL)")
    ax.set_ylabel("out-of-fold p")

    ax = axes[1, 0]
    group_scatter(ax, positives, "z_dwi_b1000", "p")
    ax.set_xlabel("lesion mean dwi (z within brain mask)")
    ax.set_ylabel("out-of-fold p")

    ax = axes[0, 2]
    group_scatter(ax, rows, "brain_ml", "p")
    ax.set_xlabel("brain-mask volume (mL)")
    ax.set_ylabel("out-of-fold p")

    ax = axes[1, 1]
    group_scatter(ax, rows, "brain_ml", "dwi_p999")
    ax.set_xlabel("brain-mask volume (mL)")
    ax.set_ylabel("dwi 99.9th pct (z)")

    ax = axes[1, 2]
    sizes = 10 + 40 * np.log10(1 + np.array([r["volume_ml"] for r in positives]) * 10)
    scatter = ax.scatter(
        [r["x_mm"] for r in positives],
        [r["z_mm"] for r in positives],
        s=sizes,
        c=[r["p"] for r in positives],
        cmap="viridis",
    )
    for record in positives:
        ax.annotate(
            record["subject"].removeprefix("sub-"),
            (record["x_mm"], record["z_mm"]),
            fontsize=7,
            xytext=(6, 0),
            textcoords="offset points",
        )
    ax.axvline(0, ls=":", c="k", lw=0.8)
    ax.set_xlabel("lesion centroid x from brain centre (mm, + = right)")
    ax.set_ylabel("lesion centroid z (mm, + = superior)")
    fig.colorbar(scatter, ax=ax, label="out-of-fold p")

    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    fig.suptitle(f"Task 1 out-of-fold scores (git {sha.stdout.strip()})")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figures/scores.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def write_table(rows: list[dict]) -> None:
    header = [
        "subject",
        "label",
        "p",
        "fourth",
        "shape",
        "in_plane_mm",
        "slice_mm",
        "fov_x",
        "fov_z",
        "brain_ml",
        "dwi_p99",
        "dwi_p999",
        "dwi_above_z3",
        "dwi_skew",
        "volume_ml",
        "voxels",
        "n_slices",
        "n_components",
        "largest_frac",
        "retained",
        "in_mask",
        "z_dwi_b1000",
        "z_adc",
        "x_mm",
        "y_mm",
        "z_mm",
        "x_frac",
        "y_frac",
        "z_frac",
    ]
    lines = ["\t".join(header)]
    for record in rows:
        values = []
        for key in header:
            value = record.get(key, "")
            values.append(f"{value:.3f}" if isinstance(value, float) else str(value))
        lines.append("\t".join(values))
    (OUT_DIR / "explore.tsv").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    (OUT_DIR / "figures").mkdir(exist_ok=True)
    rows = measure()
    dumb_baselines(rows)
    figure_scores(rows)
    figure_lesions(rows)
    figure_subjects(rows)
    write_table(rows)


if __name__ == "__main__":
    main()
