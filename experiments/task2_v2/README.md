# task2_v2

First scored run of the progressive-decoder head (PR #48) under the frozen task-2 protocol, on both
checkpoints. `0` and `1` ran at `c69d725`, before `leave_one_out` saved per-fold predictions; `2`
and `3` repeat them at the same seeds at `27e43b2`, so each pair is a replicate. Figures are drawn
at the **global** cut, not the per-subject oracle cut task 4's are — the fold carries the whole
probability volume.

```bash
sbatch launch.sh                      # array 0-3, ~18 min a run
uv run python collect.py              # the two tables below
uv run python figure_curves.py        # -> figures/curves.png
uv run python figure_predictions.py   # -> figures/<run>_predictions.png
uv run python figure_probability.py   # -> figures/<run>_probability.png
```

## Results

| run | ckpt | Dice | 95% CI | oracle | cut | zeros | paired vs baseline head | min |
|---|---|---|---|---|---|---|---|---|
| ckpt-ptfull | pt-full | **0.281** | [0.158, 0.413] | 0.329 | 0.257 | 10/23 | +0.086 [-0.006, +0.183] | 18 |
| ckpt-ptfull_folds | pt-full | **0.286** | [0.159, 0.418] | 0.326 | 0.257 | 10/23 | +0.091 [-0.000, +0.181] | 18 |
| ckpt-walnut | walnut | **0.270** | [0.138, 0.410] | 0.343 | 0.401 | 13/23 | +0.075 [+0.007, +0.143] | 17 |
| ckpt-walnut_folds | walnut | **0.277** | [0.149, 0.408] | 0.345 | 0.401 | 12/23 | +0.082 [+0.034, +0.137] | 17 |

| twin pair | mean Dice | selected cut | per-subject sd |
|---|---|---|---|
| ckpt-ptfull / ckpt-ptfull_folds | 0.281 / 0.286 | 0.2571 / 0.2571 | 0.035 |
| ckpt-walnut / ckpt-walnut_folds | 0.270 / 0.277 | 0.4012 / 0.4012 | 0.072 |

Baseline head is 0.195 with 12/23 zeros at pt-full, so only the pt-full rows are a clean A/B.

## What it says

**It reproduces.** Twin means differ by 0.005-0.007 and both pairs select the identical cut,
against per-subject swings of 0.035-0.072. Seeding is complete; the residual is CUDA, where
trilinear `interpolate` and conv3d backward both accumulate with atomics.

**The checkpoints tie**, 0.284 against 0.273 averaging twins, about twice the twin spread — but
walnut finds fewer subjects (12-13 zeros against 10) and delineates them better (oracle 0.345).

**The zeros are a detection failure, not a threshold failure.** Nine of the ten score average
precision 0.00-0.03 against a prevalence of 2.1e-4, so no cut recovers them. The exception is
sub-01, AP 0.250 at Dice 0.000: it claims 14701 voxels confidently in the wrong place, where the
baseline scored 0.494.
