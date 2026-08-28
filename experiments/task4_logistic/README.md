# task4_logistic

Ridge fits sub-cell label fractions by squared error, which the 99.8% of background cells dominate
at a prevalence of ~2e-3. Cross-entropy weights the rare positive cells by `1/p(1-p)` instead. This
swaps the head for a multi-output logistic regression on the same soft targets, over both
checkpoints, at `2eb1685`.

```bash
sbatch launch.sh                        # 8 runs: 2 heads x 2 ckpts, logistic at 1e2/1e4/1e6
sbatch launch_1.sh                      # 6 more, walking alpha down to 1e-1
uv run python collect.py                # the table below
uv run python figure_alphas.py          # -> figures/alphas.png
uv run python figure_curves.py --run <run>
uv run python figure_predictions.py --run <run>
```

Two protocol changes came in with the head, both in `2571747`/`536415b`. `THRESHOLDS` lost its
bottom three decades, which no run had ever selected into, so the same 60 points now step x1.111
rather than x1.249. And the held-out subject is scored from its cached embedding rather than
re-embedded, which is the same frozen-backbone tokens either way. Neither is why logistic wins:
`ridge_ptfull` re-measures the old `s4_c4_d04` config under both and lands at 0.256 against 0.252.

## Results

| run | head | ckpt | alpha | Dice | 95% CI | nerve | vessel | oracle | paired vs ridge/pt-full |
|---|---|---|---|---|---|---|---|---|---|
| logistic_walnut_1e1 | logistic | walnut | 1e1 | **0.355** | 0.302 – 0.402 | 0.434 | 0.277 | 0.385 | +0.099 [+0.081, +0.117] |
| logistic_walnut_1e2 | logistic | walnut | 1e2 | 0.338 | 0.286 – 0.387 | 0.402 | 0.274 | 0.370 | +0.082 [+0.067, +0.096] |
| logistic_ptfull_1e1 | logistic | pt-full | 1e1 | 0.339 | 0.283 – 0.387 | 0.401 | 0.276 | 0.370 | +0.082 [+0.064, +0.098] |
| logistic_walnut_1e0 | logistic | walnut | 1e0 | 0.337 | 0.289 – 0.379 | 0.420 | 0.253 | 0.359 | +0.080 [+0.062, +0.098] |
| logistic_ptfull_1e0 | logistic | pt-full | 1e0 | 0.328 | 0.276 – 0.373 | 0.395 | 0.260 | 0.354 | +0.071 [+0.053, +0.089] |
| logistic_ptfull_1e2 | logistic | pt-full | 1e2 | 0.308 | 0.253 – 0.359 | 0.366 | 0.251 | 0.342 | +0.052 [+0.036, +0.066] |
| logistic_walnut_1e-1 | logistic | walnut | 1e-1 | 0.304 | 0.261 – 0.342 | 0.383 | 0.226 | 0.324 | +0.048 [+0.029, +0.067] |
| logistic_ptfull_1e-1 | logistic | pt-full | 1e-1 | 0.295 | 0.250 – 0.335 | 0.359 | 0.230 | 0.318 | +0.038 [+0.019, +0.057] |
| ridge_walnut | ridge | walnut | — | 0.274 | 0.228 – 0.320 | 0.330 | 0.218 | 0.300 | +0.018 [+0.008, +0.028] |
| ridge_ptfull | ridge | pt-full | — | 0.256 | 0.211 – 0.301 | 0.317 | 0.196 | 0.283 | — |
| logistic_walnut_1e4 | logistic | walnut | 1e4 | 0.166 | 0.135 – 0.197 | 0.190 | 0.142 | 0.192 | -0.090 [-0.109, -0.070] |
| logistic_ptfull_1e4 | logistic | pt-full | 1e4 | 0.097 | 0.075 – 0.118 | 0.092 | 0.102 | 0.130 | -0.159 [-0.189, -0.130] |
| logistic_*_1e6 | logistic | both | 1e6 | 0.000 | — | 0.000 | 0.000 | 0.000 | -0.256 [-0.301, -0.211] |

## What it says

**The head is worth +0.082, and the checkpoint +0.017.** Each reproduces under the other: the head
gives +0.082 [+0.064, +0.098] on pt-full and +0.081 [+0.065, +0.096] on walnut, the checkpoint
+0.018 under ridge and +0.017 under logistic. Two accidental replicates agreeing to 0.001, which is
better evidence than either interval alone. 39 of 40 subjects improve.

**Alpha is now the dominant knob, and 1e1 is a real interior peak** — beating 1e0 by
+0.019 [+0.012, +0.026] and 1e2 by +0.017 [+0.007, +0.027]. This is the opposite of ridge, where
five decades moved CV MSE by 1-2.5% and justified dropping the internal CV. Logistic falls to 0.000
by 1e6, so the fixed alpha now needs a guard rail rather than a default.

**Logistic's threshold curve is a broad hump where ridge's is a spike** (`figures/alphas.png`,
lower left). Ridge peaks at 2e-2 and falls off on both sides; logistic holds within a factor of
three of its peak. A cut fit on 40 subjects has to transfer to the test set, so this matters beyond
the headline number, and it is invisible in the table.

**Both heads fail on the same subjects.** sub-20 scores 0.005 under ridge and 0.016 under logistic,
and `figures/*_predictions.png` shows both scattering false positives over a visibly
lower-contrast scan. That is a data problem the head cannot reach.

NSD is still unimplemented and is half the challenge rank, so this ranking is provisional in the
same way earlier task 4 tables were. The vessel remains the weaker label at 0.277 against the
nerve's 0.434, and over-claiming is what NSD punishes and Dice forgives.
