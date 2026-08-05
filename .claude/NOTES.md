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

**3. Win rate uses marginal, not paired, CIs (2026-07-29).** `analysis_table.py` scores a win when
a model's point estimate clears the opponent's bootstrap CI upper bound. A paired bootstrap
(resample subjects once, recompute both models, CI the difference) is far more powerful — 34 of 90
model-pair-task comparisons came back inconclusive under the current rule. Parked deliberately:
Connor's counter is that the marginal CI measures how a score bounces under random reconstructions
of the benchmark cohort, which may be the more relevant variance for a benchmark.
*Blocker:* `probe_global.py` writes only summary metrics to `metrics.jsonl`, so no paired test is
computable from existing `output/` dirs. Saving `y` and the repeat-averaged out-of-fold vector per
run is a couple of lines, but needs a re-run of the sweep.

**4. sMRI MAE axis order (2026-07-29).** `models/smri_mae.py` has a `transpose` flag and a TODO.
Evidence says native RAS `(X, Y, Z)` is right: padding the MNI template to 208x240x208 and dicing
against a stored pretraining brain mask gives 0.89/0.85 for identity vs 0.75 for the `(Z, Y, X)`
swap. The sweep agrees — `vitl_fomo300_tr` is the worst config in the table. Left in place pending
Connor's own check.

**5. FOMO task-4 class label order is a guess (2026-07-23).** `("nerve", "vessel")`, with a TODO in
`tasks/fomo.py`. Per-class metric names depend on it. Confirm against the challenge data.

**6. `dense_embed` contract needs a rethink (2026-07-29).** Three of five backbones declined to
implement it — Neuro-JEPA (768-d per voxel is ~24GB on the task-4 grid), SynthSeg (`postprocess`
never resamples back to the input grid), sMRI MAE. A coarse-grid + lazy-resample contract would
probably fit all three; the current per-voxel-on-the-canonical-grid one fits none.

**7. Seg probe is untuned (2026-07-27).** `NEG_PER_SUBJECT=10_000` and real-data peak memory were
deferred on "measure first" and no real backbone has run the seg probe yet. Also `_predict`
zero-fills columns for classes a fold never saw — silent degradation, suspect it first if task-4
per-class AP looks wrong.

**8. Head extraction for submission (2026-07-27).** The fitted StandardScaler + linear head is
trivially serializable; wire it up when prepping a real submission.

**9. `tasks/ppmi.py` loads via an `hf://` glob (2026-07-30).** Workaround, with a TODO: a stray
`eval/cache-*.arrow` indices file is committed alongside the shards upstream, and the repo-id
loader picks it up and dies casting to `{'indices': uint64}`. Revert to a plain `load_dataset` when
the upload is fixed.

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

- **Where the plot style preference lives.** Internal plots should be plain matplotlib — default
  colors, no annotations or custom palettes, short readable code — and only get the full treatment
  when the figure is meant to be published. This was a real correction (2026-07-30) but is not
  written down anywhere in the harness now. Covered in spirit by "keep it simple"; say if you want
  it stated explicitly.
- **`~/.claude/skills/polish` is now duplicated** by `.claude/skills/polish`. Delete the global
  copy so there is one, unless other projects depend on it.
