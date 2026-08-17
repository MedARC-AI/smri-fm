"""Rank (pooling x post-transform) pairs for tasks 6 and 7 on real features.

The challenge withholds the tasks 6/7 labels and fits its own probes, so there
is no way to score those tasks locally. What we can do is score the vector they
receive on the one local cohort with the n to support group-wise numbers: task
3's 494 subjects, which carry real age.

That gives a structural analogue of both halves of the submission:

    task 6 proxy      how well the embedding supports the task at all (r, MAE)
    task 7 proxy      the spread of that skill across age bins, which is what
                      `FairnessScore = mean_v (1 - [max_g M_g - min_g M_g])`
                      penalises, computed over the same bin edges the official
                      `pipeline/config.py` example uses

It is a proxy, not the metric. Task 3 is regression where the challenge probes
classification, and its age bins are the local cohort's, not the eval set's. A
pooling that wins here is a candidate worth a submission slot, not a proven
gain. Read it for ranking and for large effects, not for third decimals.

The expensive step is the encoder, so `cache_pooled.py` runs it once per subject
and applies every pooling inside that pass, writing only the pooled vectors --
raw ViT-L/patch-8 tokens are ~83MB per subject, 41GB over task 3, where the
pooled cache is ~30MB. Everything after that is numpy and runs on a laptop. Baseline for the current submission (`mean` + `identity`) on
walnut-v0.1 vitl/sub-52k is MAE 3.50, r 0.968, age-bin MAE spread 3.30y.

    python -m fomo_tune.bench_task7 --sandbox              # no model/data/GPU
    python -m fomo_tune.cache_pooled --out pooled.npz      # one fwd pass, on GPU
    python -m fomo_tune.bench_task7 --cache pooled.npz     # the grid, anywhere
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fomo_tune import poolings, posttransform

logger = logging.getLogger("fomo_tune")

#: The official `pipeline/config.py` example bins, so the local spread is
#: computed over the same edges the challenge's own template suggests.
AGE_BINS = ((25, "<=25"), (50, "26-50"), (75, "51-75"), (10_000, "76+"))

#: A bin below this many subjects is reported but excluded from the spread.
#: At n<12 the max-min of a per-group metric is dominated by sampling noise:
#: with an identical generating process in every group, simulated disparity
#: runs ~0.09 at n=9 against ~0.03 at n=300, so a small bin manufactures a
#: difference that has nothing to do with the embedding.
MIN_BIN_N = 12


def age_bin(age: float) -> str:
    return next(name for upper, name in AGE_BINS if age <= upper)


def group_spread(age: np.ndarray, pred: np.ndarray) -> tuple[float, dict]:
    """Max-min of per-bin MAE, the local stand-in for the challenge's D_v."""
    labels = np.array([age_bin(a) for a in age])
    per_bin, used = {}, []
    for _, name in AGE_BINS:
        m = labels == name
        if not m.any():
            continue
        per_bin[name] = {"n": int(m.sum()), "mae": float(np.abs(pred[m] - age[m]).mean())}
        if m.sum() >= MIN_BIN_N:
            used.append(per_bin[name]["mae"])
    spread = float(max(used) - min(used)) if len(used) > 1 else float("nan")
    return spread, per_bin


def evaluate(X: np.ndarray, age: np.ndarray, n_folds: int = 20, seed: int = 0) -> dict:
    """Task 3's own protocol: 20-fold, pooled out-of-fold predictions.

    The post-transform is fitted inside each fold on the training embeddings
    only. Fitting it once on everything would leak the test subjects' geometry
    into the projection and flatter every variant that uses one.
    """
    n_folds = max(2, min(n_folds, len(age)))
    oof = np.zeros(len(age))
    for train, test in KFold(n_folds, shuffle=True, random_state=seed).split(X):
        head = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 6, 19)))
        head.fit(X[train], age[train])
        oof[test] = head.predict(X[test])
    spread, per_bin = group_spread(age, oof)
    return {
        "mae": float(np.abs(oof - age).mean()),
        "pearson_r": float(np.corrcoef(age, oof)[0, 1]),
        "age_bin_mae_spread": spread,
        "per_bin": per_bin,
        "dim": int(X.shape[1]),
    }


def run_grid(tokens: list, masks: list, age: np.ndarray,
             poolings_to_run=None, transforms_to_run=None) -> list[dict]:
    """Pool in memory then bench. Only for the sandbox and small cohorts: raw
    tokens are ~83MB per subject at ViT-L/patch-8, so the real path pools during
    the forward pass and benches the cache (`run_grid_pooled`)."""
    pool_names = poolings_to_run or list(poolings.VARIANTS)
    pooled_by_name = {}
    for pname in pool_names:
        try:
            pooled_by_name[pname] = np.stack(
                [poolings.apply(pname, t, m) for t, m in zip(tokens, masks)])
        except Exception as exc:
            logger.warning(f"pooling {pname} failed: {type(exc).__name__}: {exc}")
    return run_grid_pooled(pooled_by_name, age, transforms_to_run)


def run_grid_pooled(pooled_by_name: dict, age: np.ndarray,
                    transforms_to_run=None) -> list[dict]:
    """Bench a cache of already-pooled embeddings: {pooling_name: (n, D)}."""
    rows = []
    tf_names = transforms_to_run or list(posttransform.VARIANTS)

    for pname, pooled in pooled_by_name.items():
        if pooled.shape[0] != len(age):
            raise ValueError(f"{pname}: {pooled.shape[0]} embeddings for {len(age)} subjects")

        for tname in tf_names:
            try:
                # Fitted per fold inside evaluate(); here only to fail fast on a
                # transform that cannot apply to this pooling's width at all.
                posttransform.fit(tname, pooled[: max(len(pooled) // 2, 2)])
            except Exception as exc:
                logger.warning(f"{pname}+{tname} unusable: {type(exc).__name__}: {exc}")
                continue
            rows.append({"pooling": pname, "transform": tname,
                         **evaluate_with_transform(pooled, age, tname)})
            logger.info(f"  {pname:16s} {tname:16s} "
                        f"mae {rows[-1]['mae']:.3f}  spread {rows[-1]['age_bin_mae_spread']:.2f}")
    return rows


def evaluate_with_transform(pooled: np.ndarray, age: np.ndarray,
                            tname: str, n_folds: int = 20, seed: int = 0) -> dict:
    # Clamp to the cohort. KFold raises when n_splits exceeds n_samples, and a
    # smoke run over a handful of subjects is a legitimate use of this: the
    # point there is that the plumbing runs on real embedding widths, not that
    # the numbers mean anything.
    n_folds = max(2, min(n_folds, len(age)))
    oof = np.zeros(len(age))
    for train, test in KFold(n_folds, shuffle=True, random_state=seed).split(pooled):
        f = posttransform.fit(tname, pooled[train])
        head = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 6, 19)))
        head.fit(f(pooled[train]), age[train])
        oof[test] = head.predict(f(pooled[test]))
    spread, per_bin = group_spread(age, oof)
    dim = posttransform.fit(tname, pooled)(pooled[:1]).shape[1]
    return {"mae": float(np.abs(oof - age).mean()),
            "pearson_r": float(np.corrcoef(age, oof)[0, 1]),
            "age_bin_mae_spread": spread, "per_bin": per_bin, "dim": int(dim),
            # Per-subject out-of-fold predictions. Every variant is scored on
            # the same 494 subjects in the same fold split, so keeping these
            # makes the comparison against the baseline paired. Unpaired CIs
            # overlapping settles nothing, which is the same trap the walnut
            # table carries.
            "oof": [float(v) for v in oof]}


def report(rows: list[dict]) -> str:
    rows = sorted(rows, key=lambda r: r["mae"])
    out = [f"{'pooling':16s} {'transform':16s} {'dim':>5s} {'MAE':>7s} {'r':>7s} {'spread':>7s}",
           "-" * 64]
    for r in rows:
        s = r["age_bin_mae_spread"]
        out.append(f"{r['pooling']:16s} {r['transform']:16s} {r['dim']:5d} "
                   f"{r['mae']:7.3f} {r['pearson_r']:7.4f} "
                   f"{'  n/a' if s != s else f'{s:7.2f}'}")
    return "\n".join(out)


def main_sandbox() -> int:
    """Self-test on synthetic tokens: verifies every pooling and transform runs,
    is deterministic, and keeps a fixed width across subjects. Proves the
    plumbing, says nothing about which variant is better."""
    rng = np.random.default_rng(0)
    n, n_tok, d = 120, 200, 64
    age = rng.uniform(20, 85, n)
    tokens, masks = [], []
    for a in age:
        t = rng.normal(0, 1, (n_tok, d))
        t[:5] += (a - 50) / 20.0          # a planted, focal age signal
        m = np.ones(n_tok, dtype=bool); m[rng.random(n_tok) < 0.1] = False
        tokens.append(t); masks.append(m)

    for name in poolings.VARIANTS:
        widths = {poolings.apply(name, t, m).shape[0] for t, m in zip(tokens[:5], masks[:5])}
        assert len(widths) == 1, f"{name} returned varying width {widths}"
        a = poolings.apply(name, tokens[0], masks[0])
        assert np.array_equal(a, poolings.apply(name, tokens[0], masks[0])), f"{name} not deterministic"
        assert np.isfinite(a).all(), f"{name} produced non-finite values"
    print(f"OK  {len(poolings.VARIANTS)} poolings: fixed width, deterministic, finite")

    pooled = np.stack([poolings.apply("mean", t, m) for t, m in zip(tokens, masks)])
    for name in posttransform.VARIANTS:
        f = posttransform.fit(name, pooled[:60])
        out = f(pooled[60:])
        assert out.shape[0] == 60 and np.isfinite(out).all(), name
        assert out.dtype == np.float32, f"{name} must ship float32, got {out.dtype}"
    print(f"OK  {len(posttransform.VARIANTS)} transforms: apply to held-out, float32, finite")

    rows = run_grid(tokens, masks, age,
                    poolings_to_run=["mean", "topk_mean_p05"],
                    transforms_to_run=["identity", "pca_64"])
    assert len(rows) == 4
    print("OK  grid runs end to end\n")
    print(report(rows))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sandbox", action="store_true",
                    help="self-test on synthetic tokens; no model, data or GPU")
    ap.add_argument("--cache", type=Path,
                    help="npz from cache_pooled.py: one (n, D) array per pooling, plus age")
    ap.add_argument("--out", type=Path, help="write the full grid as JSON")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.sandbox:
        return main_sandbox()
    if not args.cache:
        ap.error("pass --cache <tokens.npz>, or --sandbox to self-test")

    blob = np.load(args.cache, allow_pickle=True)
    age = blob["age"]
    # Only the 2-D (n, D) arrays are poolings; `age` and `subject` ride along
    # in the same npz and would otherwise be benched as if they were embeddings.
    meta = {"age", "subject"}
    pooled_by_name = {k: blob[k] for k in blob.files
                      if k not in meta and blob[k].ndim == 2}
    if not pooled_by_name:
        raise SystemExit(f"{args.cache} holds no (n, D) pooling arrays")
    logger.info(f"{len(age)} subjects, {len(pooled_by_name)} cached poolings: "
                + ", ".join(f"{k}({v.shape[1]}d)" for k, v in pooled_by_name.items()))
    rows = run_grid_pooled(pooled_by_name, age)
    print("\n" + report(rows))
    if args.out:
        args.out.write_text(json.dumps(rows, indent=2))
        logger.info(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
