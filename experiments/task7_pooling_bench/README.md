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

## Result

Run on walnut-v0.1 vitl/sub-52k over task 3's 494 subjects, 87 variants, one
A100, about eight minutes to encode.

**Keep `mean` + `identity`. Nothing earned a submission slot.**

The harness reproduces the team's own number exactly, which is what makes the
rest of the table worth reading: `mean` + `identity` gives MAE 3.505 and
r 0.9680 against the committed `fomo_tune_walnut_v0_1` values of 3.505 and
0.9680, on the same 494 ages.

Paired against the incumbent on the same subjects and folds, 4000 bootstrap
resamples:

| pooling | transform | dMAE | 95% CI | dspread | 95% CI |
|---|---|---|---|---|---|
| logsumexp_t1 | identity | -0.004 | [-0.021, +0.012] | -0.01 | [-0.14, +0.06] |
| mean | l2 | -0.004 | [-0.010, +0.003] | -0.00 | [-0.02, +0.04] |
| mean | pca_128 | **+0.289** | [+0.138, +0.448] | -0.28 | [-1.19, +0.38] |
| mean_topk_p05 | pca_256 | **+0.449** | [+0.254, +0.646] | -0.38 | [-1.58, +0.34] |
| mean | pca_128_drop2 | **+7.592** | [+7.043, +8.144] | **+7.70** | [+5.53, +11.06] |

**0 of 87 beat the incumbent, and no spread interval excludes zero.** Three
variants tie, none win. Every dimensionality reduction costs real MAE for a
spread change that is indistinguishable from noise, which is the tradeoff this
bench existed to measure rather than assume.

### Correction: the first conclusion was blind to scale and offset

"Nothing beats the incumbent" held only for transforms the bench could see, and
it could not see this class at all. The head above is `StandardScaler` +
`RidgeCV`, which z-scores every dimension before the probe runs, so any
transform whose effect is per-axis scale or offset was erased. Measured
directly: that head returns an identical 3.5050 MAE on features multiplied by
100, while a probe without a standardiser goes from 6.89 to 1.3e14 on the same
input.

This matters because **fomo-lp does not normalise**. `embedding_dataset.py`
states "No image transforms are applied, embeddings are already final
features", and `identity_model.py` is a pass-through, so the challenge's linear
head receives the shipped vector exactly as written, offset included.

And the offset is large. Across every pooling the embedding cloud sits about
**nine times further from the origin than its own between-subject spread**,
12.6x for `mean_std`. The variation carrying all the signal is a much smaller
effect than a constant carrying none, so a head trained by SGD for 20 epochs
spends much of its budget with the bias absorbing an offset instead of fitting
the directions that matter.

Under a probe that does not standardise, paired over the same 494 subjects:

| pooling | dMAE (centered - raw) | dspread |
|---|---|---|
| mean | **-3.225** [-3.711, -2.783] | -1.54 [-4.39, +0.13] |
| gem_p3 | **-2.819** [-3.201, -2.454] | -0.83 [-2.09, +1.12] |
| logsumexp_t1 | **-3.027** [-3.403, -2.630] | -1.09 [-2.40, +0.96] |

Every MAE interval excludes zero; every spread moves the right way without
reaching significance. The effect does not depend on the pooling, which is what
you expect from something driven by the geometry of the cloud rather than by
how it was reduced.

**Centering is downside-free in a way the other transforms are not.** A linear
head with a bias represents exactly the same function on centered or uncentered
input, so the fitted function class is unchanged and only the conditioning
differs. Worst case is neutral. PCA and whitening discard or rescale
information and can genuinely lose; centering cannot.

Two honest limits. The SGD stand-in here is weaker than the challenge's probe,
which uses momentum, a cosine schedule and a val-selected lr sweep, and a
better optimiser copes better with an offset, so **expect a smaller gain than
these numbers, not this size**. And full whitening is catastrophic (MAE 23.4,
r 0.17): whitening 1024 dimensions from ~470 samples amplifies near-null
directions that are pure noise. Centering is the safe member of this family,
not the whole family.

### The post-transform idea is refuted

Removing leading principal components destroys the task: MAE goes 3.51 to 11.1
and r goes 0.968 to 0.70 at `drop_top=2`, worse at 5. So the top variance
directions of this encoder **carry the age signal**, they are not the nuisance
directions that all-but-the-top removes in the NLP setting the idea came from.
That was the stated risk when the axis was proposed, and it is now settled on
real features rather than argued.

PCA without dropping is milder but still costs MAE for nothing measurable, so
the whole second axis comes back negative. Worth knowing: it is the kind of
plausible, literature-backed move that would have looked reasonable in a
submission and cost accuracy for no fairness gain.

### What did not move

Order statistics lose to the mean on this task, which is consistent with brain
age being a global property rather than a focal one. `max` and
`topk_mean_p01` are the worst non-degenerate variants at MAE 6.13 and 4.52.
That is not evidence about task 1, where the signal is one small infarct and
the argument for keeping the peak is much stronger.

### Caveats

This is task 3 regression standing in for a classification probe the challenge
runs on withheld labels, and these are the local cohort's age bins, not the
eval set's. It ranks candidates and catches large effects. It cannot tell you
the actual task 6 or 7 score.

`MIN_BIN_N=12` drops the 76+ bin at n=10, so the spread column here is over
three bins and reads 0.68 where the same predictions across all four bins give
3.30. The exclusion is deliberate, since at n=10 the bin is mostly sampling
noise, but the two numbers are not comparable.

### Age bias in the shipped model, unchanged by any of this

From the committed walnut task 3 predictions, per age bin MAE is 2.99, 3.54,
3.66, 6.29 with signed error +1.08, +0.39, -0.73, -4.89. Residual against
chronological age is r = -0.234, p = 1.5e-07 over all 494, the standard brain
age regression to the mean rather than the n=10 bin alone.

Out-of-fold Beheshti/de Lange bias correction, fitted per fold so no test
subject's chronological age is used, makes MAE significantly worse
(+0.137y [+0.050, +0.227]) for a spread change that is not significant
(-0.79y [-1.69, +0.39]). Not worth a slot either.

## Status

Everything here is verified only by `--sandbox`, which checks that each pooling
returns a fixed width, is deterministic and finite, and that each transform
applies to held-out data as float32. The real grid needs one GPU pass over task
3 and has not been run — no GPU on this machine.

## Running it on Narval

The team's `launch.sh` targets their own cluster. Narval needs a different
shape because its compute nodes have no internet, and both the checkpoint and
`Task_3.zip` are fetched over the network by default.

Two hatches already exist in the code, so no loader needed changing:
`open_zip` opens a path that exists instead of downloading, so pointing
`FOMO_EVAL_BASE_URL` at a local directory keeps task 3 on disk; and
`resolve_ckpt` returns a non-`hf://` path unchanged, so a prefetched `.pth`
works directly.

```bash
git clone https://github.com/saman-rahbar/smri-fm.git   # ~39MB, no submodule needed
cd smri-fm && git checkout feat/task7-fairness-harness

bash experiments/task7_pooling_bench/prefetch_narval.sh     # LOGIN node
sbatch --account=def-<supervisor> \
       experiments/task7_pooling_bench/launch_narval.sh     # compute node
```

The account is passed at submission rather than hardcoded, so a supervisor's
name is not committed to a repo that may go upstream. It must be `--account`
or `SBATCH_ACCOUNT`; `SLURM_ACCOUNT` is an output variable Slurm sets inside
the job and is ignored at submission time. The job
self-tests, smoke-runs 8 subjects before committing to all 494, then writes
`output/grid.txt`.

Narval specifics folded in, each of which has cost a job before: the venv lives
in `$PROJECT` because `$SCRATCH` is purged after ~60 days idle; `--gres` rather
than `--gpus-per-node`; `--mem` is system RAM, not VRAM; `HF_HOME` is not
exported in a fresh login shell even after a successful `huggingface-cli
login`, so it is set explicitly; `HF_HUB_DISABLE_XET=1` because Xet transfers
fail there; and `scipy-stack` on StdEnv/2023 carries no scikit-learn, which is
why the prefetch builds a `virtualenv --no-download`. The bench does not need
`asparagus`, `matplotlib` or most of `pyproject`, only the import set of
`cache_pooled`/`bench_task7` plus what `backbone.py` pulls in.

## Shipping the winner

`main_task6_and_7.py` now takes `pooling` and `transform`. **Defaults are
`mean` + `identity`, which is byte-identical to what it shipped before**, so
nothing changes until a variant is chosen deliberately.

```bash
# identity / l2 need no fit
python -m fomo_tune.main_task6_and_7 export transform=l2

# anything fitted takes the pooled cache the bench already produced; the
# projection is estimated once, offline, and frozen into the container, since
# the challenge hands predict.py one image at a time
python -m fomo_tune.main_task6_and_7 export \
    pooling=mean transform=pca_128 \
    fit_cache=experiments/task7_pooling_bench/output/pooled_walnut_v0_1.npz
```

`save` writes `post.npz` beside `config.yaml` and `load` restores it. Three
guards, because each of these would otherwise surface inside the container at
submission time: an unknown pooling or transform name is rejected at
construction; a config asking for a fitted transform whose `post.npz` is
missing refuses to load rather than silently shipping the raw pooling; and a
transform fitted on a different pooling's width is caught in `predict` with the
two widths named.

## Deciding whether a variant is worth a slot

The grid reports point estimates. Three submission slots per task per track is
not enough to spend one on noise, and the trap is already visible in the walnut
table, where the CIs are unpaired so overlapping intervals settle nothing.

Every variant is scored on the same subjects under the same fold split, so
`bench_task7` records per-subject out-of-fold predictions and the comparison
can be paired:

Both of these are numpy and sklearn over the cached npz, so they run on a login
node. The venv alone is not enough there: numpy and scipy come from
`scipy-stack`, which is only on the path once the modules are loaded, so a bare
`source .../activate` gives `ModuleNotFoundError: No module named 'numpy'`.

```bash
module load StdEnv/2023 gcc/12.3 arrow/21.0.0 python/3.11
source "${PROJECT}/fomo_task7_venv/bin/activate"
export PYTHONPATH="${PWD}/src"

python -m fomo_tune.bench_task7 \
    --cache experiments/task7_pooling_bench/output/pooled_walnut_v0_1.npz \
    --out   experiments/task7_pooling_bench/output/grid.json

python -m fomo_tune.compare_task7 \
    --grid  experiments/task7_pooling_bench/output/grid.json \
    --cache experiments/task7_pooling_bench/output/pooled_walnut_v0_1.npz
```

It reports, against `mean` + `identity`, the paired difference in MAE (the
task 6 proxy) and in age-bin spread (the task 7 proxy), each with a 95%
bootstrap interval, and a verdict per row. **The incumbent wins ties.** A
variant earns a slot only if its MAE interval excludes zero, or its spread
interval excludes zero while MAE is not made worse.

Re-running the grid is cheap — it is numpy over the cached npz — so the GPU
pass never has to be repeated to get these.

## When the GPU queue is longer than the job

The pass is 494 forward passes with no backward pass, so it does not need an
A100, it just finishes sooner on one. Three levers, cheapest first:

**Right-size the ask.** Walltime picks the backfill bucket. A 3h request sat a
full day behind what a 1h request gets, for a job that needs well under an
hour. `launch_narval.sh` now asks 1h / 4 cores / 24G, which is what it uses.

**Submit the CPU twin as well.** `launch_narval_cpu.sh` runs the same thing on
16 cores with an 11h walltime. It is slower per subject, but the CPU queue is
usually minutes against a day for a GPU, so it often finishes first in wall
time. Submit both and cancel whichever loses.

```bash
sbatch --account=def-<supervisor> experiments/task7_pooling_bench/launch_narval.sh
sbatch --account=def-<supervisor> experiments/task7_pooling_bench/launch_narval_cpu.sh
squeue -u $USER --start
```

Both write the same `output/pooled_walnut_v0_1.npz`. That is deliberate: the
cache is resumable, so if one is killed the other picks up where it stopped
rather than starting over. Do not run them at the same time on purpose.

**Another cluster.** The same RAP usually covers Rorqual, Fir, Beluga and
Cedar, and their GPU queues differ a lot hour to hour. The cost is redoing the
prefetch there, which is a 3.9GB checkpoint and a 2GB zip.

`cache_pooled` flushes every `--flush-every` subjects (default 50) through a
temp file and an atomic replace, and a re-run skips whatever is already cached.
A preemption or a walltime kill costs the subjects since the last flush, not
the pass, which is what makes the CPU route and a short walltime safe.
