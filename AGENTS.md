# AGENTS.md

Orientation for coding agents. `README.md` and `src/fomo_tune/README.md` are written for humans and
cover install, data layout, the task table, and how to run and package things — this file does not
repeat them. It carries only what you cannot get by reading the code.

**Read `CODING_STANDARDS.md` before writing any code.** It is the file most likely to contradict
your defaults: duplication is preferred over the wrong abstraction, defensive error handling and
silent fallbacks are removed on sight, everything is public, and explanatory comments get deleted.

## What this is

A structural MRI foundation model targeting the FOMO26 challenge. Two packages, not equally active:

- `src/smri_mae/` — MAE pretraining. Stable; we consume its checkpoints rather than change it.
- `src/fomo_tune/` — per-task adapters on the frozen encoder. **The active work.**

`main_task<k>.py` is one challenge task end to end, split into a tunable `Task<k>Method` and a
**frozen protocol** — the CV split, the metric, the bootstrap. Tuning means editing the method.
Editing the protocol makes the resulting score incomparable to every score already recorded, so it
is a deliberate decision rather than a step in an experiment.

`predict` is the challenge's own contract, and every cross-validation fold calls it, so the scored
path is the submitted path. Keep it that way when adding a task: fold work belongs in the method,
not in a training-only branch.

The task mains are **deliberate forks of one skeleton**, not an abstraction waiting to happen.
Hoisting their shared `features` / `fit` / `save` / `load` into a base class is the most tempting
wrong move in this repo, and `CODING_STANDARDS.md` rules it out explicitly.

## Things that cost an hour if you don't know them

**The encoder cannot run on CPU at all.** `smri_mae`'s blocks use nested-tensor SDPA, and its
backend selection queries the cuDNN attention backend — on a CUDA-built torch with no visible
device that *raises* rather than falling back. So there is no local smoke test of a forward pass.
Verify everything else on CPU, and build a depth-0 encoder to exercise the surrounding code.

**The fold draw is a bigger effect than most changes you will try.** At n=48 the k-fold shuffle seed
alone moved AUROC by 0.036, which is why the split seed is frozen at 0 independent of `cfg.seed`.
The bootstrap CIs resample subjects, not splits, so they do not contain that variance. Treat small
deltas as noise, and don't report a win you cannot separate from the fold draw.

**The git history predates the project.** `fomo_tune` was split out of a parent repo; everything
below `5d62f4d` ("reset repo") is that parent's pretraining and eval history, and the pre-split
branches are still on the remotes (`origin/nanobrain`, `origin/evals`, and others). `git log` on a
file will often tell you about work that no longer exists here.

## Current state

`.claude/NOTES.md` holds the open threads, caveats on results, and what is deliberately parked. It
is gitignored, so it is present in a working copy and absent from a fresh clone. Read it if it is
there; it is more current than this file.
