# fomo_tune

The five FOMO26 challenge tasks, one script each, tuned independently.

## Layout

| File | |
|---|---|
| `main_task<k>.py` |  One task, end to end. Each has a **frozen** "protocol" section. Anything outside of the protocol is fair game. |
| `datasets.py` | **frozen**. One `load_fomo_task<k>()` per task, streaming the challenge zips into an HF dataset. Raw niftis, no resampling — the backbone transform does that. |
| `backbone.py` | **frozen**. Frozen sMRI MAE encoder; the transform canonicalizes to RAS, rescales to 1mm, fits to the pretraining shape, z-scores in a mean-threshold brain mask. |
| `utils.py` | **frozen**. Seeding, git sha, logging. |
| `build.py` + `Apptainer.def` | **frozen**. Package a run dir into the challenge `.sif`. Shared by every task. |

**VERY IMPORTANT: The only thing you are allowed to change is the fit/predict logic inside `main_task<k>.py`. The "protocol" logic and all other utils are frozen.**

## Run

```bash
uv run python -m fomo_tune.main_task1 train
uv run python -m fomo_tune.main_task1 predict --dwi dwi.nii.gz --output prob.txt \
    --model-dir experiments/fomo_tune_baseline/output/task1/model
```

## Submit

`build.py` packages a run dir into the `.sif` the challenge wants:

```bash
uv run python -m fomo_tune.build experiments/fomo_tune_baseline/output/task1
```

### Validating

`third_party/container-validator` is the challenge's own validator, test niftis included:

```bash
python third_party/container-validator/container_validator/validate.py \
    --task task1 --sif experiments/fomo_tune_baseline/output/task1/task1.sif
```

It runs `python /app/predict.py --flair /input/… --output /output/<sid>.txt` inside an `apptainer
instance` with `/input`, `/output` and `/tmp` bound — exactly the shim's contract, so nothing in
`predict.py` is guessing at the interface. **All three containers pass**: 20 tests for task 1, 13
for task 5, 12 for task 3. The counts differ because the regression suite drops the
probability-range check, which is worth knowing — nothing the validator runs constrains task 3's
output beyond finiteness, so a badly wrong head would still pass.

One thing easy to miss: it takes GPU via `--nvccli` rather than `--nv`, and one of its tests runs
`nvidia-smi -L` **inside** the container. `python:3.11-slim` ships no `nvidia-smi`, so that test
passes only because `--nvccli` injects the host one — a CUDA base image would hide that dependency
rather than remove it.

**Build on the login node, run under `salloc --nodelist=n-6`.** Those are the only two hosts with
apptainer, and only the latter has a driver — see
`.claude/memory/sif-builds-need-apptainers-apparmor-profile.md`. A driver-less host does not fall
back to CPU; `predict` dies at the forward pass.

## What changes per task

| Task | n | Inputs | Output | Split | Notes |
|---|---|---|---|---|---|
| 1 infarct | 21 | adc, dwi_b1000, flair (+t2s/swi) | probability | LOO | done |
| 5 polymicrogyria | 48 | t1w | probability | 20-fold | done |
| 3 brain age | 494 | t1w | age in years | 20-fold | done — RidgeCV head, **Pearson r and MAE**, each with its own bootstrap CI |
| 2 meningioma | 23 | dwi_b1000, flair (+t2s/swi) | mask, input grid | LOO | drafted — flair only, per-subject **Dice** |
| 4 trigeminal | 40 | t2w | mask, labels 1=nerve 2=vessel | — | tabled |

## Gotchas

**Volumes are wildly anisotropic.** Task 1's DWI is 0.46×0.46×**5.6**mm, so the transform
upsamples z by 5.6× to reach 1mm iso. Nothing is wrong, but don't read the 1mm grid as real
resolution.

**The backbone never saw skull or neck.** Pretraining used a SynthSeg brain mask; the transform
substitutes a mean-intensity threshold, which keeps both.

**Probabilities are not calibrated.** `LogisticRegressionCV` on ~20 samples × 1024 features shrinks
hard; task 1's out-of-fold probabilities all land in 0.48–0.52 with near-perfect ranking. Fine for
AUROC, which is what the challenge scores, but don't read them as probabilities. Task 5's do span
0–1, which is n=48 rather than n=21 and not evidence of calibration.

**n is tiny.** Task 1's is ~0.06 wide at the top of the range. Most tuning deltas you chase will be inside it.
