"""What is task 5's AUROC 0.984 made of?

Tasks 1 and 3 were audited by pairing the model's out-of-fold score with scalars that need no
backbone at all. This does the same for task 5, over three groups: what the nifti header says,
what the scan covers, and what SynthSeg measures. Each scalar gets an AUROC with a
label-permutation p, groups get a leave-one-out logistic head, and the model's ranking is
rank-residualized against whatever separates.

Needs `segment.py` to have run.

    uv run python experiments/explore_fomo_task5/explore.py \
        | tee experiments/explore_fomo_task5/output/explore.log
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import rankdata

from segment import TASK_DIR, subjects

# SynthSeg's FreeSurfer labels, by tissue.
CORTEX = (3, 42)
WHITE = (2, 41)
VENTRICLE = (4, 43, 14, 15)

GROUPS = {
    "header": ["fov_ap_mm", "vox_ap", "n_slices", "matrix"],
    "coverage": ["brain_at_ap_edge", "ap_margin_mm", "brain_ml"],
    "synthseg": ["cortex_ml", "cortex_frac", "gm_wm_ratio", "ventricle_ml", "folding"],
}


def scalars(sub: str, out: Path) -> dict:
    img = nib.load(TASK_DIR / "preprocessed" / sub / "ses_01/t1.nii.gz")
    zooms = img.header.get_zooms()
    seg = np.asarray(nib.load(out / sub / "seg.nii.gz").dataobj)

    brain = seg > 0
    cortex = np.isin(seg, CORTEX)
    # 6-neighbour boundary of the cortex label, normalized so a sphere of any size scores the
    # same. Polymicrogyria is many small gyri, so it should show up as extra surface per volume.
    faces = sum(
        (cortex ^ np.roll(cortex, shift, axis)).sum() for axis in range(3) for shift in (1, -1)
    )
    # anterior-posterior coverage: how much brain sits in the outermost slice of the scan, and
    # how far the nearest brain voxel is from the edge of the field of view
    ap_profile = brain.sum(axis=(0, 2))
    live = np.flatnonzero(ap_profile)
    edge = max(ap_profile[0], ap_profile[-1])

    return {
        "subject": sub,
        "n_slices": img.shape[1],
        "vox_ap": zooms[1],
        "fov_ap_mm": img.shape[1] * zooms[1],
        "matrix": img.shape[0],
        "brain_ml": brain.sum() / 1e3,
        "cortex_ml": cortex.sum() / 1e3,
        "ventricle_ml": np.isin(seg, VENTRICLE).sum() / 1e3,
        "cortex_frac": cortex.sum() / brain.sum(),
        "gm_wm_ratio": cortex.sum() / np.isin(seg, WHITE).sum(),
        "folding": faces / cortex.sum() ** (2 / 3),
        "brain_at_ap_edge": edge / ap_profile.max(),
        "ap_margin_mm": min(live[0], len(ap_profile) - 1 - live[-1]),
    }


def auroc_with_perm(y: np.ndarray, value: np.ndarray, rng, n_perm: int = 10000) -> tuple:
    """AUROC of one scalar, and the two-sided permutation p of |AUROC - 0.5|."""
    observed = roc_auc_score(y, value)
    null = [roc_auc_score(rng.permutation(y), value) for _ in range(n_perm)]
    p = (np.abs(np.array(null) - 0.5) >= abs(observed - 0.5)).mean()
    return observed, p


def loo_auroc(X: np.ndarray, y: np.ndarray) -> float:
    """Leave-one-out logistic head, so a group of scalars is scored the way the model is."""
    oof = np.zeros(len(y))
    for train, test in LeaveOneOut().split(X):
        head = make_pipeline(
            StandardScaler(),
            LogisticRegressionCV(
                Cs=10,
                class_weight="balanced",
                max_iter=2000,
                l1_ratios=(0,),
                use_legacy_attributes=False,
            ),
        )
        head.fit(X[train], y[train])
        oof[test] = head.predict_proba(X[test])[:, list(head.classes_).index(1)]
    return roc_auc_score(y, oof)


def main() -> None:
    parser = argparse.ArgumentParser()
    here = Path(__file__).parent
    parser.add_argument("--out", type=Path, default=here / "output")
    parser.add_argument("--figures", type=Path, default=here / "figures")
    parser.add_argument(
        "--preds",
        type=Path,
        default=here.parents[0] / "fomo_tune_walnut_v0_1/output/task5/preds.json",
        help="a run's out-of-fold predictions, for the residualization",
    )
    args = parser.parse_args()
    rng = np.random.default_rng(0)

    labels = dict(subjects())
    table = pd.DataFrame([scalars(sub, args.out) for sub in labels])
    table["label"] = [labels[sub] for sub in table["subject"]]
    table.to_csv(here / "explore.tsv", sep="\t", index=False, float_format="%.4f")
    y = table["label"].to_numpy()

    columns = [c for group in GROUPS.values() for c in group]
    print(f"\n{'scalar':<18} {'AUROC':>7} {'perm p':>8}   control mean / case mean")
    rows = []
    for column in columns:
        value = table[column].to_numpy(dtype=float)
        auroc, p = auroc_with_perm(y, value, rng)
        rows.append((column, auroc, p))
        print(
            f"{column:<18} {auroc:>7.3f} {p:>8.4f}   "
            f"{value[y == 0].mean():.3g} / {value[y == 1].mean():.3g}"
        )

    print(f"\n{'features':<28} {'LOO AUROC':>9}")
    for name, group in GROUPS.items():
        print(f"{name:<28} {loo_auroc(table[group].to_numpy(dtype=float), y):>9.3f}")
    print(f"{'all scalars':<28} {loo_auroc(table[columns].to_numpy(dtype=float), y):>9.3f}")

    if args.preds.exists():
        rows_json = map(json.loads, args.preds.read_text().splitlines())
        preds = {r["subject"]: r["pred"] for r in rows_json}
        model = np.array([preds[sub] for sub in table["subject"]])
        print(f"\nmodel ({args.preds.parents[2].name}) AUROC {roc_auc_score(y, model):.3f}")
        for column, _, _ in sorted(rows, key=lambda r: -abs(r[1] - 0.5))[:3]:
            value = table[column].to_numpy(dtype=float)
            residual = rankdata(model) - np.polyval(
                np.polyfit(rankdata(value), rankdata(model), 1), rankdata(value)
            )
            print(
                f"  residualized against {column:<18} AUROC {roc_auc_score(y, residual):.3f}"
                f"  (spearman {np.corrcoef(rankdata(model), rankdata(value))[0, 1]:+.3f})"
            )
    else:
        model = None

    panels = list(dict.fromkeys([max(rows, key=lambda r: abs(r[1] - 0.5))[0], "fov_ap_mm"]))
    panels += ["folding", "ventricle_ml"]
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 3.6))
    for ax, column in zip(axes, panels):
        for label, marker in ((0, "o"), (1, "s")):
            keep = y == label
            ax.scatter(
                table[column][keep],
                model[keep] if model is not None else rng.normal(0, 0.05, keep.sum()),
                marker=marker,
                label="PMG" if label else "control",
            )
        ax.set_xlabel(column)
        ax.set_ylabel("model p" if model is not None else "jitter")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.figures / "scalars.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
