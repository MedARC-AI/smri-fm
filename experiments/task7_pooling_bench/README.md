# task7_pooling_bench

A bench for the tasks 6/7 embedding, plus what the official scoring code turns
out to reward.

Tasks 6 and 7 ship one artifact: a container mapping `nifti -> (D,) float32`.
The challenge fits its own probes on withheld labels, so nothing about those
tasks can be scored locally. `main_task6_and_7.py` currently ships a mean over
tokens, and that pooling is the entire tunable surface.

## What the scoring actually rewards

Read off [fomo-lp](https://github.com/fomo26/fomo-lp) and
[fomo-metrics](https://github.com/fomo26/fomo-metrics):

```
FairnessScore(M) = (1/|V'|) * sum_v (1 - D_v)
D_v              = max_g M(group g) - min_g M(group g)
```

`M` is `ovr_f1` or `ovr_auroc` **of the task 6 probe**, computed within each
demographic group. Task 7 is the spread of task 6's own metric across groups.

Three consequences, each verified by running their code:

**It is accuracy parity, not attribute leakage.** Nothing here measures whether
age or sex is decodable from the embedding. Debiasing methods that strip a
protected direction (INLP and friends) are not what this rewards, and can widen
the spread by hurting whichever group depended on the direction removed.

**Small groups dominate through noise.** With an identical generating process in
every group and zero real unfairness, simulated disparity runs 0.091 at n=9
against 0.027 at n=300. The fomo-lp README's own example has a group at n=17 of
300. A meaningful part of the score is sampling noise nobody controls, which is
why `MIN_BIN_N` in `bench_task7.py` excludes tiny bins rather than reporting
their spread as if it meant something.

**`compute_ovr_auroc` raises whenever exactly 2 classes are present in a group**
(sklearn wants 1-D scores in the binary case), and `compute_max_disparity`
catches every exception and drops that group silently. If one group survives,
disparity is `0.0` and the variable contributes a perfect `1.0`. On a binary
probe task the AUROC half of the fairness score does not run at all. This looks
like a bug in the official package rather than a design choice and is worth
raising with the organisers.

The useful corollary: task 7 penalises the spread of the metric task 6
maximises, so **raising the worst group is the only direction that helps both**.
Lowering the ceiling would trade one task for the other.

## The bench

The expensive step is the encoder and every candidate embedding is a different
reduction of the same tokens, so `cache_pooled.py` runs the backbone once per
subject and applies every pooling inside that pass. Raw ViT-L/patch-8 tokens are
~83MB per subject (41GB over task 3); the pooled cache is ~30MB and everything
downstream runs on a laptop.

```bash
python -m fomo_tune.bench_task7 --sandbox            # no model, data or GPU
python -m fomo_tune.cache_pooled --out pooled.npz    # one forward pass, GPU
python -m fomo_tune.bench_task7 --cache pooled.npz --out grid.json
```

Two axes:

- **`poolings.py`** — mean (incumbent), max, top-k mean, GeM, logsumexp, and two
  concatenations. Order statistics are in because task 1's signal is one small
  infarct out of thousands of tokens, and the mean divides it by N.
- **`posttransform.py`** — a fixed map applied to the pooled vector, fitted on
  *unlabeled* embeddings: L2, PCA to a target width, whitening, and dropping
  leading components. Legal under Methods track, one matmul at inference.

The post-transform is the axis nobody has tried. Motivation is the NLP
anisotropy literature (Mu & Viswanath's all-but-the-top, ICLR 2018; the
BERT-whitening line), where frozen transformer embeddings are dominated by a few
high-variance nuisance directions and removing or rescaling them reliably
improves linear probes. The brain-MRI analogue of word frequency is intensity,
head size and site, which fits the read that tasks 1 and 5 sit at ceiling on
"brain mask volume + brightness as proxy".

Scored on task 3's 494 subjects, the only local cohort with the n to support
group-wise numbers: `r`/`MAE` stand in for task 6, and max-min of per-age-bin
MAE stands in for task 7, over the bin edges the official `config.py` example
uses. It is a proxy — task 3 is regression where the challenge probes
classification, and these are not the eval set's bins. Read it for ranking and
large effects, not third decimals.

## What is established so far

**Simulation, against a faithful copy of the challenge probe** (multi-head SGD,
lr sweep, val selection, 20 epochs), 60 paired seeds at D=1024, n_train=80/class:

| transform | disparity vs raw | probe macro F1 vs raw |
|---|---|---|
| PCA 256 | +0.000 [-0.016, +0.017] | +0.003 [-0.004, +0.010] |
| PCA 128 | **-0.023 [-0.039, -0.006]** | **+0.034 [+0.025, +0.042]** |

Both CIs exclude zero only at 128. So the mechanism is real and moves tasks 6
and 7 the same way, but it needs aggressive reduction and the effect is modest.

An earlier version of this table showed a far larger effect. That was an
artifact of standing in for the probe with unregularised sklearn logistic
regression, which collapses in p>>n where their SGD-plus-val-selection probe
does not. The number above is the corrected one.

**This is simulation, not evidence about the walnut embeddings.** It assumes the
dominant variance directions carry no class signal. In the sandbox, where the
planted signal *is* in the top components, `drop_top` variants destroy the task
(r goes negative). Which case the real encoder is in is exactly what the bench
settles, and it is why `drop_top` is in the grid rather than in the submission.

**Age bias in the current submission, from the committed walnut-v0.1 task 3
predictions** (`experiments/fomo_tune_walnut_v0_1/output/task3/preds.json`):

| age bin | n | MAE | mean signed error |
|---|---|---|---|
| <=25 | 128 | 2.99 | +1.08 |
| 26-50 | 142 | 3.54 | +0.39 |
| 51-75 | 214 | 3.66 | -0.73 |
| 76+ | 10 | 6.29 | -4.89 |

Spread 3.30y, 95% CI [1.17, 5.54]. Residual against chronological age is
r = -0.234, p = 1.5e-07 over all 494, so this is the standard brain-age
regression to the mean, not the n=10 bin alone.

**The obvious fix does not work.** Out-of-fold Beheshti/de Lange age-bias
correction, fitted per fold so no test subject's chronological age is used:
MAE +0.137y [+0.050, +0.227] (significantly worse) for a spread change of
-0.79y [-1.69, +0.39] (not significant). Not worth a submission slot. Recorded
so nobody spends a day rediscovering it.

## Status

Everything here is verified only by `--sandbox`, which checks that each pooling
returns a fixed width, is deterministic and finite, and that each transform
applies to held-out data as float32. The real grid needs one GPU pass over task
3 and has not been run — no GPU on this machine.
