# fomo_tune

The five FOMO26 challenge tasks, one script each, tuned independently.

## Layout

| File | |
|---|---|
| `main_task<k>.py` | One task, end to end. Each has a **frozen** "protocol" section. Anything outside of the protocol is fair game. |
| `datasets.py` | **frozen**. One `load_fomo_task<k>()` per task, streaming the challenge zips into an HF dataset. Raw niftis, no resampling — the backbone transform does that. |
| `backbone.py` | **frozen**. Frozen sMRI MAE encoder; the transform canonicalizes to RAS, rescales to 1mm, fits to the pretraining shape, z-scores in a mean-threshold brain mask. |
| `utils.py` | **frozen**. Seeding, git sha, logging. |
| `build.py` + `Apptainer.def` | **frozen**. Package a run dir into the challenge `.sif`. Shared by every task. |

## Pre-requisites

Download the FOMO eval data for local exploration

```bash
uvx hf download medarc/smri-fm \
    --include 'fomo_eval/*' \
    --local-dir ./data \
    --repo-type dataset
unzip 'data/fomo_eval/*.zip' -d data/fomo_eval/
```

> nb, on the cluster these data are at `/data/smri-datasets/fomo_eval/`.

## Run

```bash
uv run python -m fomo_tune.main_task1 train
uv run python -m fomo_tune.main_task1 predict \
    --adc data/fomo_eval/Task_1/preprocessed/sub-01/ses-01/adc.nii.gz \
    --dwi data/fomo_eval/Task_1/preprocessed/sub-01/ses-01/dwi_b1000.nii.gz \
    --flair data/fomo_eval/Task_1/preprocessed/sub-01/ses-01/flair.nii.gz \
    --output prob.txt \
    --model-dir output/fomo_tune/task1/model
```

## Submit

Make sure you have [apptainer](https://apptainer.org/docs/admin/main/installation.html#install-from-pre-built-packages) installed.

`build.py` packages a run dir into the `.sif` the challenge wants:

```bash
uv run python -m fomo_tune.build output/fomo_tune/task1
```

> nb, this is just for reference. you probably don't need to worry about building the containers.

### Validating

`third_party/container-validator` is the challenge's own validator, test niftis included:

```bash
uv run python third_party/container-validator/container_validator/validate.py \
    --task task1 --sif output/fomo_tune/task1/task1.sif
```

## What changes per task

| Task | n | Inputs | Output | Split | Notes |
|---|---|---|---|---|---|
| 1 infarct | 21 | adc, dwi_b1000, flair (+t2s/swi) | probability | LOO | done |
| 2 meningioma | 23 | dwi_b1000, flair (+t2s/swi) | mask, input grid | LOO | drafted — flair only, per-subject **Dice** |
| 3 brain age | 494 | t1w | age in years | 20-fold | done — RidgeCV head, **Pearson r and MAE**, each with its own bootstrap CI |
| 4 trigeminal | 40 | t2w | mask, labels 1=nerve 2=vessel | — | tabled |
| 5 polymicrogyria | 48 | t1w | probability | 20-fold | done |
| 6+7 probing, fairness | — | one image, any modality | 1024-d embedding `.npy` | — | drafted — no labels and no head, so `export` in place of `train` |

## Gotchas

**Volumes are wildly anisotropic.** Task 1's in-plane spacing runs 0.44–0.90mm against a slice
thickness of **5.2–7.2**mm (median 6.5), so the transform upsamples z by ~6× to reach 1mm iso.
Nothing is wrong, but don't read the 1mm grid as real resolution.

**The backbone never saw skull or neck.** Pretraining used a SynthSeg brain mask; the transform
substitutes a mean-intensity threshold, which keeps both.

**Probabilities are not calibrated.** `LogisticRegressionCV` on ~20 samples × 1024 features shrinks
hard; task 1's out-of-fold probabilities all land in 0.48–0.52 with near-perfect ranking. Fine for
AUROC, which is what the challenge scores, but don't read them as probabilities. Task 5's do span
0–1, which is n=48 rather than n=21 and not evidence of calibration.

**n is tiny.** Task 1's AUROC CI is ~0.06 wide at the top of the range. Most tuning deltas you chase will be inside it.
