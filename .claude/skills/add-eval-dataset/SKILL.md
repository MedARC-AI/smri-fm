---
name: add-eval-dataset
description: Add a benchmark dataset and its tasks to the nanobrain eval suite. Use when porting a new cohort — measuring confounds, matching cohorts, and choosing which splits are worth shipping.
scope: project
---

# Adding an eval dataset

Settled over CNP, ABIDE, ADHD-200 and PPMI-mini. The dataset-side counterpart to `add-eval-model`.
Per-dataset findings are in `.claude/memory/` — see `MEMORY.md` under "Datasets and HF".

## The rule that drives everything

A probe on frozen features will happily read scanner, site or age instead of the biology. **An
AUROC that is really site is worse than no benchmark.** So the confound work happens *before* the
task is written, not after a result looks surprising.

Note that a dumb baseline and a confound floor are different checks, and the first can pass while
the second fails. On PPMI-mini, random features were at chance *and* scan year alone read PD vs CN
at 0.61. Chance-level baselines tell you the model isn't fooling you; nuisance floors tell you the
benchmark isn't.

"Model AUROC minus confound AUROC" is not a fix. AUROCs do not subtract.

## 1. Survey the phenotype table

Report before writing the task:

- what candidate labels exist and their group sizes
- what acquisition handles exist — site, scanner, field strength, scan date
- if there is no scanner column, what proxies exist. **Scan year** and **header geometry**
  (shape + zooms) have both worked; PPMI-mini needed both, since the cohorts were enrolled in waves.

Traps: acquisition-QC flags can be confounded themselves, so check before filtering on them; and
phenotype tables carry duplicate subject rows — ADHD-200 lists 101 subjects twice.

## 2. Measure each nuisance floor on its own

Cross-validated logistic probe on the nuisance variable alone, per candidate task. Report those
numbers before writing anything. This is the output that decides which tasks ship.

## 3. Match, and expect it to cost subjects

Match cases to controls within (site/scanner, age band, sex) cells using
`tasks/utils.py:matched_indices`. Connor consistently spends subjects to kill a shortcut — he
accepted 265 -> 82 on one CNP task, and PPMI PD vs CN cost 583 subjects to get 426 with every floor
back at chance. Prefer matching over documenting a confound and living with it. He does not want IQ
matched.

Match coarsely: year bands were enough on PPMI, where exact-year or geometry-class matching cost
130-140 more subjects and bought nothing once the floors were at chance.

## 4. Reject what can't be fixed

Some splits are unfixably confounded and should not ship. Say so and give the number: PPMI's
Prodromal vs CN had a matched pool of 146 that still left geometry at 0.57, because CN is
2010-2013 and prodromal is almost all 2018+, so the two barely overlap in time.

Don't add a fourth task measuring what three existing tasks already measure.

## 5. Report

Post-matching floors per confound, plus a `random_features` baseline, alongside the shipped task
sizes. Then register with `@register_task` and write the rejected splits, with their floors, to a
memory file under `.claude/memory/`.

## Cache gotcha

Editing anything under `tasks/` does not invalidate the HF generator cache, so a rerun silently
loads data built by the old code. Purge it first — see the gotchas in
[src/nanobrain/eval/README.md](../../../src/nanobrain/eval/README.md).
