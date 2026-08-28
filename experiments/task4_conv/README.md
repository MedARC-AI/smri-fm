# task4_conv

First scored runs of `main_task4_conv.py`: task 2's grid of progressive CNN decoders on task 4's
geometry, features and protocol. Both checkpoints, each at two seeds.

```bash
sbatch launch.sh                # array 0-3, ~50 min a run (100 on a contended node)
uv run python collect.py        # the table below
uv run python figure_curves.py  # -> figures/<run>_curves.png
```

## Results

All four runs at `2eb1685`. `collect.py` rebuilds this from `output/*/`.

| run | ckpt | seed | Dice | 95% CI | nerve | vessel | oracle | cuts | vs ridge | min |
|---|---|---|---|---|---|---|---|---|---|---|
| ckpt-ptfull | pt-full | 4466 | **0.133** | [0.106, 0.162] | 0.146 | 0.120 | 0.167 | 6.0e-01/6.7e-01 | -0.123 [-0.158, -0.091] | 49 |
| ckpt-ptfull_s2 | pt-full | 1234 | **0.136** | [0.110, 0.160] | 0.150 | 0.121 | 0.168 | 6.0e-01/6.7e-01 | -0.120 [-0.154, -0.087] | 49 |
| ckpt-walnut | walnut | 4466 | **0.149** | [0.121, 0.177] | 0.173 | 0.125 | 0.186 | 6.0e-01/6.7e-01 | -0.126 [-0.159, -0.093] | 100 |
| ckpt-walnut_s2 | walnut | 1234 | **0.160** | [0.132, 0.189] | 0.183 | 0.136 | 0.195 | 6.0e-01/6.7e-01 | -0.115 [-0.148, -0.081] | 100 |

Ridge at the same scale, subcell and depth: **0.256** pt-full, **0.274** walnut.

## What it says

**The conv head loses, by about half the ridge's Dice**, -0.115 to -0.126 paired, every CI clear of
zero. It reproduces: the seed pairs differ by 0.003 and 0.011 against a per-subject sd of 0.05-0.06.
The two heads agree on which subjects are hard (r = 0.67 per subject), so this is not a plumbing
failure. Where task 2 gained +0.086 from this swap, task 4 loses 0.12.

**It is not a thresholding failure.** The oracle cut is 0.186 against the ridge's 0.300, so the
scores themselves carry less.

**The failure is a diffuse vessel and a cut that will not generalize.** At the selected cuts the
head claims 7.5x the vessel's true voxels (median, up to 37x) where the ridge claims 1.03x, and it
misses the nerve entirely in 11 of 40 subjects against the ridge's 4. The map is a plateau rather
than a gradient: the vessel goes from 7.5x over-claimed at cut 0.60 to *nothing* at 0.955, and
subjects disagree widely about where in that band to cut (oracle 10-90th percentile 0.53-0.85,
against 0.024-0.032 for the ridge). That is the signature of a head pushed to over-predict, which
points at `pos_weight` — lifted here to 20-180 from a crop prevalence that was guessed rather than
measured, then averaged over a 9x range where task 2 spanned 4x.

**`THRESHOLDS` under-resolves this head, so the numbers above are an underestimate.** The grid
stayed geometric, which suits the ridge's ~2e-3 label fractions but puts 55 of its 60 points below
0.6, where a sigmoid head does nothing. A linear or logit-spaced grid is the fix. The oracle gap of
0.11 is far too large for grid resolution alone to explain.

### The curves

`figures/ckpt-walnut_curves.png`. Every subject's Dice climbs steeply into the global cut and falls
to zero at the next grid point. Subjects spike to 0.35-0.55 at their own cut and the spikes land on
different points, so the mean over them is a smear peaking at 0.13-0.18. The head can segment a
given subject; what it cannot do is put two subjects' probabilities on the same scale, and the
protocol scores one global cut.
