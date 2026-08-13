# explore_fomo_task3

Task 3 scores **r=0.963, MAE 3.69y** out of fold on the challenge's own 494 subjects. The
validation leaderboard scores the same container at **r=0.426, MAE 12.28y**. This asks what
accounts for the difference.

```bash
uv run python experiments/explore_fomo_task3/scan.py       # -> scan.tsv, scan.npz   (cpu, 30s)
uv run python experiments/explore_fomo_task3/embed.py      # -> oof.npz              (gpu, 6min)
uv run python experiments/explore_fomo_task3/perturb.py    # -> perturb.npz          (gpu, 45min)
uv run python experiments/explore_fomo_task3/explore.py    # -> tables, figures/
```

`embed.py` re-runs the baseline protocol and reproduces `metrics.json` to four decimals
(r=0.9631, MAE 3.6910, both CIs identical), so the embeddings and per-subject predictions it
saves are the ones the reported score was made of.

## Summary

The local score is not inflated by anything found here. It is not a leak, not duplicate subjects,
not the cohort's source structure, and not a head-size confound. What the audit does find is that
the model's output moves by **years** in response to input differences that carry no age
information at all, and that is the one mechanism consistent with a score that survives every
internal check and still falls apart on a held-out cohort.

## 1. Four explanations that are ruled out

**Not a leak in the protocol.** Folds are subject-level with one session each, and `RidgeCV`
selects alpha by its own leave-one-out inside the training fold. The in-sample/out-of-fold gap
(MAE 2.28 against 3.69) is the size a ridge on 1024 features at n=494 should cost.

**Not duplicate or repeated subjects.** Every volume was downsampled 8x and correlated against
every other. The maximum off-diagonal correlation is **0.984** against a median of 0.884, with a
smooth distribution and no gap — and the closest pairs are unrelated in age (sub-002 at 44y with
sub-165 at 57y, sub-247 at 30y with sub-079 at 68y). No subject has a partner above 0.99.

**Not a head-size confound**, which is what task 1 turned out to be. Brain-mask volume alone
scores r=0.261, and a ridge on 8x-downsampled raw voxels scores r=0.153.

**Not the cohort's source structure**, though that structure is real and worth knowing about. The
494 are several source datasets concatenated in file order: subject index correlates −0.61 with
age, mean |age(i+1) − age(i)| is 8.2y against 19.9y under a label shuffle (perm p < 1e-5), and
the blocks are visible in `figures/cohorts.png` — ids 1–191 average 61y, ids 451–494 are 44
subjects at 24.9 ± 2.1y.

Knowing *only* which contiguous block of 10 ids a subject falls in gives r=0.862 with no image at
all, so a model that recognised its source cohort would get most of the score free. It does not
need to:

| protocol | r | MAE |
|---|---|---|
| 20-fold shuffled (as shipped) | 0.963 | 3.69 |
| leave one contiguous id window out, width 25 | 0.962 | 3.76 |
| leave one contiguous id window out, width 50 | 0.961 | 3.80 |
| leave one contiguous id window out, width 100 | 0.960 | 3.83 |

Holding out 100 consecutive subjects — so that no same-source subject is ever in training — costs
**0.003 r**. Within a block of 50, the model still ranks age at r=0.916. Whatever the model reads,
it is not cohort identity.

## 2. The model-free floor is high, and legitimately so

Fifteen scalars measured off the images without a backbone, on the same 20-fold split:

| features | r | MAE |
|---|---|---|
| brain-mask volume, 1 scalar | 0.261 | 14.53 |
| dark (csf) fraction, 1 scalar | 0.620 | 11.35 |
| intensity histogram, 32 bins | 0.801 | 7.73 |
| 15 model-free scalars | 0.870 | 6.76 |
| the frozen MAE + ridge | **0.963** | **3.69** |

Unlike task 1, this floor is not a confound. The scalar carrying it is the fraction of dark
voxels inside the brain mask (spearman **+0.609** with age) — csf, so ventricular enlargement and
widened sulci. That is the actual anatomy of brain aging, and a model reading it is doing the
task. The backbone adds a real margin on top: 3.69 against 6.76 MAE.

## 3. The errors have no structure

Out-of-fold error was correlated against all nineteen measured scalars. Nothing reaches |rho| =
0.08:

| | max |rho| vs error | scalar |
|---|---|---|
| signed error | 0.071 | median intensity |
| absolute error | 0.134 | bbox y |

Brain volume, csf fraction, sharpness, bounding box, brain position in the box, intensity
percentiles, file size — all flat. The one systematic feature is the ordinary brain-age
regression to the mean: `pred = +3.05 + 0.933 × age`, so the oldest are under-aged and the
youngest over-aged by a few years. `figures/subjects.png` shows all 494 subjects sorted by age
with their errors; the large errors are not visibly different scans.

This is a clean negative. It also means the local data contains no example of the failure the
leaderboard is reporting, which is why the next section had to perturb the inputs to find one.

## 4. What the model is fragile to

`perturb.py` re-extracts features from perturbed inputs and applies the *shipped* head, so
`identity` reproduces the submitted in-sample prediction and is the control. Each row is a
difference another scanner or another preprocessing pipeline could plausibly introduce. 100
subjects spread over the age range; the cost is in years of predicted age.

| perturbation | r | MAE | mean shift | mean abs shift |
|---|---|---|---|---|
| identity | 0.988 | 2.20 | 0.00 | 0.00 |
| intensity x1.25 | 0.988 | 2.20 | −0.00 | **0.01** |
| background noise 2% | 0.988 | 2.20 | 0.01 | 0.07 |
| bias field 10% | 0.988 | 2.21 | 0.51 | 0.51 |
| flip left-right | 0.986 | 2.31 | −0.29 | 0.72 |
| shift z 10mm | 0.986 | 2.44 | −0.37 | 0.79 |
| noise in brain 5% | 0.987 | 2.50 | −0.96 | 0.98 |
| shift x 10mm | 0.983 | 2.75 | −0.99 | 1.61 |
| strip 2mm tighter | 0.982 | 3.01 | −1.05 | 1.78 |
| rotate 10 degrees | 0.981 | 3.15 | 0.42 | 1.98 |
| voxel size 1.05x | 0.971 | 3.53 | −2.04 | 2.04 |
| strip 2mm looser | 0.983 | 3.02 | −0.64 | 2.12 |
| acquired at 1.5mm | 0.982 | 5.49 | −5.02 | 5.07 |
| voxel size 0.90x | 0.982 | 5.82 | +5.49 | 5.57 |
| voxel size 1.10x | 0.980 | 8.09 | −8.07 | 8.07 |
| blur 1mm | 0.977 | 8.28 | −8.16 | 8.22 |
| acquired at 2mm | 0.978 | 8.38 | −8.24 | 8.28 |
| **voxel size 1.20x** | 0.974 | 13.11 | **−13.09** | 13.09 |
| **blur 2mm** | 0.950 | 25.44 | **−25.43** | 25.43 |
| **background noise 5%** | **0.173** | 81.12 | **−81.10** | 81.10 |

Three things come out of this.

**Intensity scaling is exactly invariant** (0.01y), which is the transform's z-scoring working as
designed, and geometric jitter is cheap — a 10mm shift, a 10 degree rotation or a left-right flip
all cost under 2y. The model is not fragile in the ways one would first guess.

**Apparent head scale is the dominant sensitivity.** Rescaling the voxel size costs roughly
**0.7 years per 1% of linear scale**: 5% is 2.0y, 10% is 8.1y, 20% is 13.1y. Nothing about the
anatomy changed — only the millimetres the header claims. This is the same finding as task 1's,
in a different task: this backbone reads absolute head size, hard. Losing genuine resolution is
comparable (a real 2mm acquisition costs 8.2y) and points the same way, since blurring and
downsampling both shrink apparent structural detail.

**There is a cliff, and it is in the token mask.** `mask = data > data.mean()` in the transform
does double duty: it normalizes, and it selects which patches become tokens, via
`patch_mask = patch_num_obs > 0` in the encoder — **one voxel above threshold validates an entire
8mm token**. The task 3 images have a hard-zeroed background, so 3121 of 20280 tokens are live.
Add background noise at 5% of the brain median and scattered speckle clears the threshold
everywhere, taking it to **14958 of 20280**, and the mean-pooled embedding is then mostly air.
The prediction moves 81 years and r falls to 0.173.

The cliff is sharp, not gradual: at 2% background noise nothing happens at all (0.07y, 3121
tokens unchanged), because the speckle has to clear `data.mean()` before any of it counts.
Whether a cohort falls off the cliff is close to binary.

## 5. Reading the leaderboard gap

**What does not explain it.** No mixture of the moderate perturbations above reproduces the
leaderboard. Sampling one condition per subject from a heterogeneous set (voxel size ±10%, 1.5mm
acquisition, 1mm blur, 10 degree rotation, looser strip) gives **r=0.937, MAE 5.14** against the
leaderboard's 0.426 and 12.28. Neither does anything else tried locally: training on a single
100-subject id block and testing on the other 394 still gives r=0.88–0.93; training only on ages
30–70 and testing outside that band gives r=0.968 with MAE 7.75. Error does not grow with
distance to the nearest training subject. **The model's ranking is remarkably hard to break.**

**The one thing that does.** Only the token-mask cliff collapses r. If it tripped on part of a
cohort rather than all of it, one free parameter gets both leaderboard numbers into range:

| fraction of subjects tripped | r | MAE |
|---|---|---|
| 0.00 | 0.988 | 2.20 |
| 0.05 | 0.679 | 6.01 |
| 0.10 | 0.527 | 10.25 |
| **0.125** | **0.492** | **11.87** |
| 0.15 | 0.442 | 14.23 |
| 0.20 | 0.395 | 17.91 |
| *leaderboard* | *0.426* | *12.28* |

The fit is close but not exact, and it is worth being precise about how close. Matching the MAE
puts the fraction at ~0.13, which predicts r=0.49 against the 0.426 observed; matching r instead
puts it at ~0.16, which predicts MAE 14.8 against 12.28. So a single tripped fraction cannot hit
both at once — it is off by roughly 15% on whichever one it is not fitted to. The leaderboard
point sits between the two, which is what a mixture of this failure plus milder effects would
look like, but that is a second free parameter and it is not evidence.

*(An earlier version of this table matched much more tightly. That run sampled its 100 subjects
wrongly and covered only ages 19–61, sd 14.4 against the cohort's 17.4; restricted range inflated
the apparent agreement. The numbers above are the corrected full-range run.)*

**This is a hypothesis, not a demonstration.** We have never seen a validation image, and the
mechanism is only established on synthetic perturbations of the training cohort. A different
story could fit the same two numbers about as well. What makes it worth acting on is that it is
the only mechanism found that collapses r at all, that it is the right order of magnitude, and
that the fix is cheap and testable locally.

## Figures

- `cohorts.png` — age against subject id in file order, with the out-of-fold error below. The
  source blocks are obvious in the top panel and invisible in the bottom one.
- `scores.png` — prediction against age, error against age, the bimodal age histogram, and error
  against the three scalars that carry the most age signal.
- `subjects.png` — all 494 subjects sorted by age, one axial slice at 55% of the brain box,
  labelled with age, subject id, brain-mask volume and out-of-fold error.
- `perturb.png` — how far each input perturbation moves the predicted age.

## What this suggests next

1. **Replace the mean-threshold mask.** It is documented as a stand-in for the SynthSeg mask used
   in pretraining, and it is the single point of failure here. Anything more robust — Otsu, a
   percentile, a largest-connected-component filter on the mask, or requiring a *fraction* of a
   patch to be observed rather than one voxel — removes the cliff. The last of these is a
   one-line change in the encoder's `patch_mask` and is worth measuring first, because it also
   bounds how much damage a partial strip failure can do.
2. **Check the tasks that share the transform.** Tasks 1, 5, 6 and 7 all pool tokens selected the
   same way. Tasks 6 and 7 ship raw embeddings, so if any challenge-side image has a live
   background the embedding is air, and there is no head to absorb it.
3. **Test on DLBS.** This audit ran out of road because there is only one cohort here, and every
   local stress test leaves r above 0.88. An actual out-of-distribution cohort is the experiment
   that would settle it. Blocked on native-space brain masks — the 25 SynthSeg outputs under
   `data/DLBS/output` are an April run that stopped, and `derivatives/masks` are MNI-space.
4. **Consider scale normalization.** 0.7 years per 1% of linear head scale is worth knowing about
   independently of the leaderboard, and it is shared with task 1's head-size confound.

## Figures

- `cohorts.png` — age against subject id in file order, with the out-of-fold error below. The
  source blocks are obvious in the top panel and invisible in the bottom one.
- `scores.png` — prediction against age, error against age, the bimodal age histogram, and error
  against the three scalars that carry the most age signal.
- `subjects.png` — all 494 subjects sorted by age, one axial slice at 55% of the brain box,
  labelled with age, subject id, brain-mask volume and out-of-fold error.
- `perturb.png` — how far each input perturbation moves the predicted age.
