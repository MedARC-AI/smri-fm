# Notes

Current project state. Prune entries when they resolve.

## Open threads

Roughly in priority order.

**1. `fomo_task1_infarct` is too small to read (2026-07-28).** n=21. Random features score
0.48 [0.19, 0.71] — the CI spans almost the whole range, and an earlier run had *both* dumb
baselines below chance (random features 0.396, random U-Net 0.225, the latter's CI excluding 0.5).
Two independent frozen backbones landing under 0.5 rather than scattering around it points at the
labels, not at noise. It currently contributes a cell to every model's aggregate. Decide whether to
fix, drop, or exclude from the aggregate.

**2. The aggregate columns average over tasks that separate nothing (2026-07-31).** ABIDE,
ADHD-200 and CNP-ADHD sit at chance for every model *including* random features, so win rate and
mean rank in `experiments/eval_global_0728/figures/table.md` are computed over roughly five
informative tasks and ten noise cells. Either weight or exclude the tasks that no model beats
chance on, or state the caveat wherever the table is shown.

**5. FOMO task-4 class label order is a guess (2026-07-23).** `("nerve", "vessel")`, with a TODO in
`tasks/fomo.py`. Per-class metric names depend on it. Confirm against the challenge data.

**7. Seg probe is untuned (2026-07-27, updated 2026-08-06).** `NEG_PER_SUBJECT=10_000` was deferred
on "measure first" and is now clearly oversized: a subject's sampled voxels collapse onto at most N
patches, so most of those 10k rows are duplicates. Also `predict_probs` zero-fills columns for
classes a fold never saw — silent degradation, suspect it first if task-4 per-class AP looks wrong.

**11. The SynthSeg wrapper wants another edit pass (2026-08-06, Connor).** Two things are not
crisp: the `resample` "verbatim" / "torch" branch, and leaning on upstream helpers for trivial
preprocessing (`find_closest_number_divisible_by_m`, `pad_volume`, `rescale_volume`,
`align_volume_to_ref`) — the last of which is also where the `patch_embed` affine has to unpick a
reorientation. The idea: **one all-in-PyTorch preprocessing pipeline, verified equivalent to the
reference**, which collapses the branch and drops the helper dependency. `resample_torch` and
`bottleneck_box` are already most of the way there, and `test_synthseg.py` already pins the
verbatim path bit-for-bit, so equivalence is testable. Worth **upstreaming to the SynthSeg-pytorch
fork** afterwards, which would give fast SynthSeg inference generally, not just in this wrapper.

**8. Head extraction for submission (2026-07-27).** The fitted StandardScaler + linear head is
trivially serializable; wire it up when prepping a real submission.

**9. `tasks/ppmi.py` loads via an `hf://` glob (2026-07-30).** Workaround, with a TODO: a stray
`eval/cache-*.arrow` indices file is committed alongside the shards upstream, and the repo-id
loader picks it up and dies casting to `{'indices': uint64}`. Revert to a plain `load_dataset` when
the upload is fixed.

**12. First seg-probe numbers are at the floor, and we don't yet know why (2026-08-06).**
`experiments/eval_seg_0806`, `random_features`, the first time any backbone has run the seg probe on
real data:

| task | n | Dice | voxel-AP |
|---|---|---|---|
| fomo_task1_infarct_seg | 21 | 0.017 | 0.027 |
| fomo_task2_meningioma | 23 | 0.012 | 0.021 |
| fomo_task4_trigeminal | 40 | 0.001 | 0.001 |

Two readings and they are not yet distinguishable. **(a) Honest floor:** `random_features` projects
16-voxel blocks, so for a structure smaller than one block the feature is mostly background, and a
random baseline pinning at prevalence is what a floor should look like. **(b) The brain mask is
eating the labels:** `voxel_targets` now restricts labels to a mean-intensity brain mask before
sampling, where previously every labelled voxel was trained on regardless. That makes train and test
agree — scoring was always in-brain — but it can cut the positive count, and a trigeminal nerve
sitting in bright CSF is exactly the case that would go sub-mean.

**Next step: measure the in-mask foreground fraction per seg task** (datasets are cached, so it is
quick). Near 1 means the numbers are honest; low means the mask is the problem. Worth settling
before the real backbones' results land on top of it. Note a class falling *entirely* out of mask
across all subjects trips the "classes absent" assert, so that end of the failure is loud.

## Caveats on results

- **ADNI numbers from before 2026-07-29 are not comparable.** `medarc/adni-mini` was re-uploaded
  that day, brain-masked and 1000 -> 1200 scans, split renamed `test` -> `eval`. The current sweep
  is post-update. See `.claude/memory/adni-mini-reuploaded-in-place.md` for details.
- **`adni_sex`, `cnp_sex` and `dlbs_sex` are wiring anchors, not results.** Every backbone lands
  >= 0.96 AUROC, so they are excluded from the table; anything much below means the wiring is
  wrong, not the model.
- **Raw-space tasks understate MNI-pretrained backbones.** CNP streams native-space T1w off
  OpenNeuro, so a head-and-neck FOV gets squeezed into a box sized for a registered brain. Fair
  across models, but it is not measuring what the task name says. Undecided whether to add
  registration as a task-side option.

## Parked

**3. Win rate uses marginal, not paired, CIs (2026-07-29).** `analysis_table.py` scores a win when
a model's point estimate clears the opponent's bootstrap CI upper bound. A paired bootstrap
(resample subjects once, recompute both models, CI the difference) is far more powerful — 34 of 90
model-pair-task comparisons came back inconclusive under the current rule. Parked deliberately:
Connor's counter is that the marginal CI measures how a score bounces under random reconstructions
of the benchmark cohort, which may be the more relevant variance for a benchmark.
*Blocker:* `probe_global.py` writes only summary metrics to `metrics.jsonl`, so no paired test is
computable from existing `output/` dirs. Saving `y` and the repeat-averaged out-of-fold vector per
run is a couple of lines, but needs a re-run of the sweep.
