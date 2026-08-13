"""What is task 3's r=0.963 made of, and why does the leaderboard say 0.426?

Reads the out-of-fold predictions and embeddings from `embed.py`, the model-free per-subject
measurements from `scan.py`, and the input-perturbation sweep from `perturb.py`. Findings are in
README.md.

    uv run python experiments/explore_fomo_task3/explore.py
"""

import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT_DIR = Path(__file__).parent
ALPHAS = np.logspace(-3, 6, 19)
COLUMNS = 26
SCALARS = (
    "brain_ml",
    "box_fill",
    "dark_frac",
    "sharpness",
    "p05",
    "p25",
    "p75",
    "p95",
    "skew",
    "kurtosis",
    "bbox_x",
    "bbox_y",
    "bbox_z",
    "median",
    "bytes",
)


def load() -> dict:
    saved = np.load(OUT_DIR / "oof.npz")
    scan = np.load(OUT_DIR / "scan.npz")
    lines = (OUT_DIR / "scan.tsv").read_text().strip().split("\n")
    header, rows = lines[0].split("\t"), [line.split("\t") for line in lines[1:]]
    subjects = list(saved["subjects"])
    assert [r[0] for r in rows] == subjects, "scan.tsv and oof.npz disagree on subjects"

    data = {
        "subjects": subjects,
        "index": np.array([int(s.split("-")[1]) for s in subjects]),
        "age": saved["age"].astype(float),
        "pred": saved["pred"],
        "embed": saved["embed"],
        "tile": scan["tile"],
        "histogram": scan["histogram"],
    }
    data |= {
        key: np.array([float(r[i]) for r in rows])
        for i, key in enumerate(header)
        if key not in ("subject", "age")
    }
    return data


def report(name: str, age: np.ndarray, pred: np.ndarray) -> dict:
    record = {
        "n": len(age),
        "r": float(np.corrcoef(age, pred)[0, 1]),
        "mae": float(np.abs(age - pred).mean()),
    }
    print(f"{name:38s} n={record['n']:3d} r={record['r']:.3f} mae={record['mae']:5.2f}", flush=True)
    return record


def fit_oof(features: np.ndarray, age: np.ndarray, cv) -> np.ndarray:
    head = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
    return cross_val_predict(head, features, age, cv=cv)


def table_baselines(data: dict) -> None:
    """What the age is worth without the backbone, on the same 20-fold split."""
    print("\n== model-free floors, same 20-fold protocol")
    age = data["age"]
    cv = KFold(20, shuffle=True, random_state=0)
    scalars = np.column_stack([data[k] for k in SCALARS])

    report("shipped out-of-fold predictions", age, data["pred"])
    report("MAE embedding, 1024-d", age, fit_oof(data["embed"], age, cv))
    report("15 model-free scalars", age, fit_oof(scalars, age, cv))
    report("intensity histogram, 32 bins", age, fit_oof(data["histogram"], age, cv))
    report(
        "histogram + scalars", age, fit_oof(np.column_stack([data["histogram"], scalars]), age, cv)
    )
    report("brain volume alone", age, fit_oof(data["brain_ml"][:, None], age, cv))
    report("dark (csf) fraction alone", age, fit_oof(data["dark_frac"][:, None], age, cv))


def table_cohorts(data: dict) -> None:
    """The 494 are several source cohorts concatenated in file order. Does the model need that?"""
    print("\n== source-cohort structure")
    age, pred, index = data["age"], data["pred"], data["index"]

    gaps = np.abs(np.diff(age)).mean()
    rng = np.random.default_rng(0)
    null = np.array([np.abs(np.diff(rng.permutation(age))).mean() for _ in range(10000)])
    print(
        f"mean |age(i+1) - age(i)| = {gaps:.2f} against {null.mean():.2f} shuffled "
        f"(perm p = {np.mean(null <= gaps):.5f}), so file order tracks source"
    )

    for width in (10, 25, 50):
        block = (index - 1) // width
        block_age = np.array([age[block == b].mean() for b in block])
        centred_age, centred_pred = age.copy(), pred.copy()
        for b in np.unique(block):
            m = block == b
            centred_age[m] -= centred_age[m].mean()
            centred_pred[m] -= centred_pred[m].mean()
        print(
            f"  blocks of {width:3d}: block mean age alone r={np.corrcoef(age, block_age)[0, 1]:.3f} "
            f"mae={np.abs(age - block_age).mean():5.2f} | "
            f"model within block r={np.corrcoef(centred_age, centred_pred)[0, 1]:.3f} "
            f"mae={np.abs(centred_age - centred_pred).mean():5.2f}"
        )

    print("\n-- leave one contiguous id window out: no same-source subject is ever in training")
    for width in (25, 50, 100):
        block = (index - 1) // width
        oof = np.zeros(len(age))
        for b in np.unique(block):
            held = block == b
            head = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
            head.fit(data["embed"][~held], age[~held])
            oof[held] = head.predict(data["embed"][held])
        report(f"leave-one-window-out, width {width}", age, oof)


def table_errors(data: dict) -> None:
    """Does anything measurable in the image predict where the model goes wrong?"""
    print("\n== error structure")
    error = data["pred"] - data["age"]
    slope, intercept = np.polyfit(data["age"], data["pred"], 1)
    print(f"pred = {intercept:+.2f} + {slope:.3f} * age  (regression to the cohort mean)")

    print(f"\n{'scalar':16s} {'rho vs age':>11s} {'rho vs error':>13s} {'rho vs |error|':>15s}")
    for key in SCALARS + ("mask_ml", "centroid_x", "centroid_y", "centroid_z"):
        v = data[key]
        print(
            f"{key:16s} {spearmanr(v, data['age']).statistic:+11.3f} "
            f"{spearmanr(v, error).statistic:+13.3f} {spearmanr(v, np.abs(error)).statistic:+15.3f}"
        )


def table_perturbations(data: dict) -> None:
    """Years of predicted age lost to differences another site's data would plausibly carry."""
    path = OUT_DIR / "perturb.npz"
    if not path.exists():
        print("\n== perturbations: perturb.npz missing, run perturb.py on a gpu")
        return

    print("\n== input perturbations, shipped head, 100 subjects")
    saved = np.load(path)
    age = saved["age"].astype(float)
    base = saved["identity"]
    names = [k for k in saved.files if k not in ("subjects", "age")]
    print(f"{'perturbation':24s} {'r':>7s} {'mae':>7s} {'mean shift':>11s} {'mean |shift|':>13s}")
    for name in names:
        v = saved[name]
        print(
            f"{name:24s} {np.corrcoef(age, v)[0, 1]:7.3f} {np.abs(age - v).mean():7.2f} "
            f"{np.mean(v - base):11.2f} {np.abs(v - base).mean():13.2f}"
        )


def table_mixture(data: dict) -> None:
    """A single perturbation shifts every subject alike, so r survives and only the mean moves.
    A cohort whose subjects differ from each other converts those shifts into scatter, which is
    the shape the leaderboard reports. The mixture below is an illustration, not a measurement:
    the real validation cohort's composition is unknown."""
    path = OUT_DIR / "perturb.npz"
    if not path.exists():
        return
    saved = np.load(path)
    age = saved["age"].astype(float)
    mixture = [
        "identity",
        "voxel_size_1.10x",
        "voxel_size_0.90x",
        "acquired_at_1.5mm",
        "blur_1mm",
        "rotate_10deg",
        "strip_looser_2mm",
    ]
    available = [name for name in mixture if name in saved.files]

    print(f"\n== a heterogeneous cohort, sampling one condition per subject from {available}")
    rng = np.random.default_rng(0)
    stacked = np.stack([saved[name] for name in available])
    scores = []
    for _ in range(200):
        pick = rng.integers(0, len(available), size=len(age))
        pred = stacked[pick, np.arange(len(age))]
        scores.append((np.corrcoef(age, pred)[0, 1], np.abs(age - pred).mean()))
    scores = np.array(scores)
    print(
        f"  r   = {scores[:, 0].mean():.3f} [{np.percentile(scores[:, 0], 5):.3f}, "
        f"{np.percentile(scores[:, 0], 95):.3f}]   leaderboard 0.426"
    )
    print(
        f"  mae = {scores[:, 1].mean():.2f} [{np.percentile(scores[:, 1], 5):.2f}, "
        f"{np.percentile(scores[:, 1], 95):.2f}]   leaderboard 12.28"
    )
    print("  -> nowhere near. no mixture of moderate perturbations collapses r.")

    print("\n== only the token-mask cliff collapses r. what if it hits part of a cohort?")
    intact, tripped = saved["identity"], saved["background_noise_5pct"]
    print(f"{'frac tripped':>13s} {'r':>7s} {'mae':>7s}")
    for fraction in (0.0, 0.05, 0.10, 0.125, 0.15, 0.20, 0.30):
        scores = []
        for _ in range(400):
            hit = rng.random(len(age)) < fraction
            pred = np.where(hit, tripped, intact)
            scores.append((np.corrcoef(age, pred)[0, 1], np.abs(age - pred).mean()))
        scores = np.array(scores)
        print(f"{fraction:13.3f} {scores[:, 0].mean():7.3f} {scores[:, 1].mean():7.2f}")
    print(f"{'leaderboard':>13s} {0.426:7.3f} {12.28:7.2f}")


def figure_subjects(data: dict) -> None:
    """One mid-axial slice per subject, sorted by age. The whole cohort on one page."""
    order = np.argsort(data["age"])
    tiles = data["tile"][order]
    height, width = tiles.shape[1:]
    rows = int(np.ceil(len(order) / COLUMNS))

    mosaic = np.zeros((rows * height, COLUMNS * width), dtype=np.uint8)
    for k, tile in enumerate(tiles):
        r, c = divmod(k, COLUMNS)
        mosaic[r * height : (r + 1) * height, c * width : (c + 1) * width] = tile

    fig, ax = plt.subplots(figsize=(COLUMNS * 0.95, rows * 1.16))
    ax.imshow(mosaic, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    for k, i in enumerate(order):
        r, c = divmod(k, COLUMNS)
        error = data["pred"][i] - data["age"][i]
        ax.text(
            c * width + 2,
            r * height + 9,
            f"{data['age'][i]:.0f}y {data['subjects'][i].removeprefix('sub-')}",
            color="yellow",
            fontsize=3.6,
        )
        ax.text(
            c * width + 2,
            r * height + height - 3,
            f"{data['brain_ml'][i]:.0f}mL  {error:+.0f}y",
            color="#7fd4ff",
            fontsize=3.6,
        )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        "Task 3, all 494 subjects sorted by age. one axial slice at 55% of the brain box.\n"
        "yellow: true age, subject id.  blue: brain-mask volume, out-of-fold error",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figures/subjects.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def figure_cohorts(data: dict) -> None:
    age, pred, index = data["age"], data["pred"], data["index"]
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax = axes[0]
    ax.scatter(index, age, s=14, label="true age")
    window = np.ones(15) / 15
    ax.plot(
        index[7:-7],
        np.convolve(age, window, mode="same")[7:-7],
        c="k",
        lw=1.2,
        label="15-subject running mean",
    )
    ax.set_ylabel("age (years)")
    ax.legend(fontsize=8)
    ax.set_title("Task 3: age against subject id, in file order")

    ax = axes[1]
    ax.scatter(index, pred - age, s=14, c="C3")
    ax.axhline(0, c="k", lw=0.8, ls=":")
    ax.set_ylabel("out-of-fold error (years)")
    ax.set_xlabel("subject id")
    for boundary in range(0, 500, 50):
        for panel in axes:
            panel.axvline(boundary + 0.5, c="0.8", lw=0.6, zorder=0)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "figures/cohorts.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def figure_scores(data: dict) -> None:
    age, pred = data["age"], data["pred"]
    error = pred - age
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    ax = axes[0, 0]
    ax.scatter(age, pred, s=16, alpha=0.7)
    limits = [age.min() - 3, age.max() + 3]
    ax.plot(limits, limits, "k:", lw=0.9, label="identity")
    slope, intercept = np.polyfit(age, pred, 1)
    ax.plot(
        limits, intercept + slope * np.array(limits), "C3", lw=1, label=f"fit, slope {slope:.3f}"
    )
    ax.set_xlabel("age (years)")
    ax.set_ylabel("out-of-fold prediction")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.scatter(age, error, s=16, alpha=0.7)
    ax.axhline(0, c="k", ls=":", lw=0.9)
    ax.set_xlabel("age (years)")
    ax.set_ylabel("error (years)")

    ax = axes[0, 2]
    ax.hist(age, bins=np.arange(18, 84, 2))
    ax.set_xlabel("age (years)")
    ax.set_ylabel("subjects")
    ax.set_title("the cohort is bimodal", fontsize=9)

    for ax, key, label in (
        (axes[1, 0], "dark_frac", "dark (csf) fraction inside the brain mask"),
        (axes[1, 1], "brain_ml", "brain-mask volume (mL)"),
        (axes[1, 2], "sharpness", "mean gradient magnitude inside the brain"),
    ):
        scatter = ax.scatter(data[key], age, s=16, c=np.abs(error), cmap="viridis")
        ax.set_xlabel(label)
        ax.set_ylabel("age (years)")
        ax.set_title(
            f"rho vs age {spearmanr(data[key], age).statistic:+.3f}, "
            f"vs |error| {spearmanr(data[key], np.abs(error)).statistic:+.3f}",
            fontsize=9,
        )
        fig.colorbar(scatter, ax=ax, label="|error| (years)")

    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    fig.suptitle(f"Task 3 out-of-fold scores (git {sha.stdout.strip()})")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figures/scores.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def figure_perturbations(data: dict) -> None:
    path = OUT_DIR / "perturb.npz"
    if not path.exists():
        return
    saved = np.load(path)
    age, base = saved["age"].astype(float), saved["identity"]
    names = [k for k in saved.files if k not in ("subjects", "age", "identity")]
    shifts = [saved[name] - base for name in names]
    order = np.argsort([np.abs(s).mean() for s in shifts])

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax = axes[0]
    ax.boxplot(
        [shifts[i] for i in order],
        vert=False,
        labels=[names[i] for i in order],
        widths=0.6,
        flierprops={"markersize": 2},
    )
    ax.axvline(0, c="k", ls=":", lw=0.9)
    ax.set_xlabel("change in predicted age (years)")
    ax.set_title("what a plausible acquisition difference costs")

    ax = axes[1]
    ax.scatter(age, base, s=18, c="k", label="identity", zorder=3)
    for i in order[-3:]:
        ax.scatter(age, saved[names[i]], s=14, alpha=0.6, label=names[i])
    ax.plot([age.min(), age.max()], [age.min(), age.max()], "k:", lw=0.9)
    ax.set_xlabel("age (years)")
    ax.set_ylabel("predicted age")
    ax.legend(fontsize=8)
    ax.set_title("the three most damaging conditions")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "figures/perturb.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    (OUT_DIR / "figures").mkdir(exist_ok=True)
    data = load()
    table_baselines(data)
    table_cohorts(data)
    table_errors(data)
    table_perturbations(data)
    table_mixture(data)
    figure_cohorts(data)
    figure_scores(data)
    figure_perturbations(data)
    figure_subjects(data)


if __name__ == "__main__":
    main()
