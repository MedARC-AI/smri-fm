"""Decide whether any variant actually beats the shipped one.

`bench_task7` reports point estimates. Three submission slots per task per
track is not enough to spend one on a difference that is noise, and the trap is
already visible in the walnut table, where the CIs are unpaired so overlapping
intervals settle nothing either way.

Every variant here is scored on the same 494 subjects under the same fold
split, so the comparison can be paired: resample subjects, recompute both
variants' metrics on that resample, and take the difference. A paired interval
is far tighter than two marginal ones and answers the question actually being
asked, which is whether this variant beats the incumbent on these subjects.

    python -m fomo_tune.compare_task7 --grid experiments/.../output/grid.json

Reports, against `mean` + `identity`:

    d_MAE       negative is better (task 6 proxy)
    d_spread    negative is better (task 7 proxy: less across-group variation)

A variant is only worth a slot if its MAE interval excludes zero, or its spread
interval excludes zero while MAE does not get worse. Anything else is a tie,
and the incumbent wins ties.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from fomo_tune.bench_task7 import AGE_BINS, MIN_BIN_N, age_bin

BASELINE = ("mean", "identity")
N_BOOT = 4000


def _spread(age: np.ndarray, pred: np.ndarray, labels: np.ndarray) -> float:
    used = []
    for _, name in AGE_BINS:
        m = labels == name
        if m.sum() >= MIN_BIN_N:
            used.append(np.abs(pred[m] - age[m]).mean())
    return float(max(used) - min(used)) if len(used) > 1 else float("nan")


def paired(age, base_oof, var_oof, n_boot=N_BOOT, seed=0):
    """Bootstrap the paired difference in MAE and in age-bin spread."""
    age = np.asarray(age, float)
    base_oof, var_oof = np.asarray(base_oof, float), np.asarray(var_oof, float)
    labels = np.array([age_bin(a) for a in age])
    rng = np.random.default_rng(seed)

    d_mae, d_spr = [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(age), len(age))
        d_mae.append(np.abs(var_oof[i] - age[i]).mean()
                     - np.abs(base_oof[i] - age[i]).mean())
        a, b = _spread(age[i], var_oof[i], labels[i]), _spread(age[i], base_oof[i], labels[i])
        if a == a and b == b:
            d_spr.append(a - b)

    def ci(d):
        if not len(d):
            return float("nan"), float("nan"), float("nan")
        d = np.asarray(d)
        lo, hi = np.percentile(d, [2.5, 97.5])
        return float(d.mean()), float(lo), float(hi)

    return ci(d_mae), ci(d_spr)


def verdict(mae_ci, spr_ci) -> str:
    """The incumbent wins ties. A variant has to earn a submission slot."""
    (m, mlo, mhi), (s, slo, shi) = mae_ci, spr_ci
    mae_better = mhi < 0
    mae_worse = mlo > 0
    spr_better = shi < 0 if shi == shi else False
    if mae_better and spr_better:
        return "BOTH better"
    if mae_better:
        return "MAE better"
    if spr_better and not mae_worse:
        return "spread better, MAE unharmed"
    if mae_worse:
        return "worse"
    return "tie"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", type=Path, required=True, help="grid.json from bench_task7")
    ap.add_argument("--cache", type=Path, help="pooled npz, for the true ages")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args()

    rows = json.loads(args.grid.read_text())
    if not rows or "oof" not in rows[0]:
        sys.exit("grid.json has no per-subject predictions; re-run bench_task7 "
                 "after pulling the commit that records them")

    if args.cache:
        age = np.load(args.cache, allow_pickle=False)["age"]
    else:
        guess = args.grid.parent / "pooled_walnut_v0_1.npz"
        if not guess.exists():
            sys.exit("pass --cache <pooled.npz>; the true ages live there")
        age = np.load(guess, allow_pickle=False)["age"]

    by_key = {(r["pooling"], r["transform"]): r for r in rows}
    if BASELINE not in by_key:
        sys.exit(f"grid has no {BASELINE} row to compare against")
    base = by_key[BASELINE]

    print(f"baseline {BASELINE[0]} + {BASELINE[1]}: "
          f"MAE {base['mae']:.3f}  r {base['pearson_r']:.4f}  "
          f"spread {base['age_bin_mae_spread']:.2f}  ({len(age)} subjects)\n")
    print(f"{'pooling':16s} {'transform':16s} {'dMAE':>7s} {'95% CI':>18s} "
          f"{'dspread':>8s} {'95% CI':>18s}  verdict")
    print("-" * 104)

    out = []
    for (pool, tf), r in by_key.items():
        if (pool, tf) == BASELINE:
            continue
        mae_ci, spr_ci = paired(age, base["oof"], r["oof"], n_boot=args.n_boot)
        v = verdict(mae_ci, spr_ci)
        out.append(((pool, tf), mae_ci, spr_ci, v))

    # Anything that beats the incumbent first, then ties, then the rest.
    rank = {"BOTH better": 0, "MAE better": 1, "spread better, MAE unharmed": 2,
            "tie": 3, "worse": 4}
    for (pool, tf), (m, mlo, mhi), (s, slo, shi), v in sorted(
            out, key=lambda x: (rank[x[3]], x[1][0])):
        sp = "     n/a           n/a  " if s != s else f"{s:8.2f} [{slo:+6.2f},{shi:+6.2f}]"
        print(f"{pool:16s} {tf:16s} {m:+7.3f} [{mlo:+6.3f},{mhi:+6.3f}] {sp}  {v}")

    winners = [o for o in out if o[3] not in ("tie", "worse")]
    print(f"\n{len(winners)} of {len(out)} variants beat the incumbent on a paired test.")
    if not winners:
        print("Keep mean + identity. Nothing here has earned a submission slot.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
