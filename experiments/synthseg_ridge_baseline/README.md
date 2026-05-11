# DLBS SynthSeg Ridge Baseline

Subject-grouped RidgeCV baseline that predicts age from SynthSeg regional volumes.
Serves as a simple, cheap reference to compare against deep-learning brain-age models.

Expected local layout from the `smri-fm` repo root:

```text
DLBS/
  synthseg/                   # SynthSeg outputs per subject/session
  participants.tsv            # DLBS participant metadata with AgeMRI_W{1,2,3}
experiments/synthseg_ridge_baseline/
```

`DLBS/` is a local input/output directory and is ignored by git.

## Run

From the repo root:

```bash
uv run python experiments/synthseg_ridge_baseline/scripts/run_ridge.py
```

Defaults:

```text
synthseg dir: DLBS/synthseg
participants: DLBS/participants.tsv
output dir:   DLBS/qc/synthseg_ridge_simple
feature set:  icv-normalized    # each region / total intracranial volume
folds:        5                  # subject-grouped CV
alphas:       0.01 .. 10000
```

Outputs in `--output-dir`:

- `predictions.csv` — per-scan out-of-fold age predictions and residuals
- `coefficients.csv` — standardized ridge coefficients from the full-data refit
- `metrics.json` — CV metrics, per-wave metrics, per-fold alpha/MAE
- `figure.png` — scatter, residuals, per-wave error, per-fold MAE, top features

## Notes

- CV is subject-grouped (`GroupKFold` on `participant_id`) so longitudinal waves
  of the same subject do not leak across folds.
- Ages are `AgeMRI_W{1,2,3}` in years, matched to each session.
- `--feature-set raw` keeps absolute volumes instead of dividing by total
  intracranial volume.
- `--min-qc FLOAT` drops scans whose per-region SynthSeg `qc_score` minimum
  falls below the threshold.
