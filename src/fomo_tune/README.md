# fomo_tune

The FOMO26 challenge tasks, one script each, tuned independently.

## Layout

| File | |
|---|---|
| `main_task<k>.py` | One task, end to end. Each has a **frozen** "protocol" section. Anything outside of the protocol is fair game. |
| `datasets.py` | **frozen**. One `load_fomo_task<k>()` per task plus the fixed DLBS development test, streaming the zips into HF datasets. Raw niftis, no resampling — the backbone transform does that. |
| `backbone.py` | **frozen**. Frozen sMRI MAE encoder; the transform canonicalizes to RAS, rescales to 1mm, fits to the pretraining shape, z-scores in a mean-threshold brain mask. |
| `utils.py` | **frozen**. Seeding, git sha, logging. |
| `build.py` + `Apptainer.def` | **frozen**. Package a run dir into the challenge `.sif`. Shared by every task. |

## Pretrained model

The pretrained model is a ViT-L MAE trained on 208x240x208 1mm volumes with patch size 8 and mask ratio 0.8. The checkpoints with original configs are on huggingface at [`medarc/walnut`](https://huggingface.co/medarc/walnut/tree/main/checkpoints). Our default checkpoint is from the [`pretrain_full_90_10_h100`](https://huggingface.co/medarc/walnut/tree/main/checkpoints/pretrain_full_90_10_h100) run ([`mihirneal/35ef89d`](https://github.com/mihirneal/smri-fm/tree/35ef89df797e0086f6cc8f5f6b9c195ae3595690)), which was trained on [FOMO300K](https://huggingface.co/datasets/FOMO-MRI/FOMO300K/tree/main) webdataset shards.

**The pretrained model is considered frozen. No new pretrain checkpoints will be accepted for the challenge.**

## Pre-requisites

If you are on the MedARC cluster, setup your environment to use the shared huggingface cache.

```bash
export HF_HOME="/data/smri-datasets/huggingface"
```

This will save re-downloading the datasets and checkpoint weights.

You can also use the data in `/data/smri-datasets` for one-off exploration. If you're not on the cluster, you can use data at [`medarc/smri-fm`](https://huggingface.co/datasets/medarc/smri-fm).

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

> Nb: @clane9 will handle final build and submission to FOMO! You can ignore this section.


Challenge submission requires building an [Apptainer](https://apptainer.org/) container image (credit: @UmerHA for working on this).

```bash
uv run python -m fomo_tune.build output/fomo_tune/task1
```

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
| 3 brain age | 494 + 128 | t1w | age in years | fit SALD, test DLBS | **Pearson r and MAE** |
| 4 trigeminal | 40 | t2w | mask, labels 1=nerve 2=vessel | — | tabled |
| 5 polymicrogyria | 48 | t1w | probability | 20-fold | done |
| 6+7 probing, fairness | — | one image, any modality | 1024-d embedding `.npy` | — | drafted — no labels and no head, so `export` in place of `train` |

## Leaderboard

### Task 1 — infarct, AUROC, LOO over 21

| Run | AUROC | 95% CI | Time | Git | Notes |
|---|---|---|---|---|---|
| baseline | 0.990 | 0.944 – 1.000 | 11s | `1df2e5d`† | dwi_b1000 only, `LogisticRegressionCV` |
| walnut-v0.1 | 0.894 | 0.731 – 1.000 | 11s | `ead1264` | vitl/sub-52k checkpoint, baseline otherwise |

### Task 2 — meningioma, Dice, LOO over 23

| Run | Dice | 95% CI | Oracle | Time | Git | Notes |
|---|---|---|---|---|---|---|
| baseline | 0.195 | 0.098 – 0.303 | 0.271 | 174s | `7d13f45` | flair only, largest-component filter, threshold 0.011 |
| no largest component | 0.170 | 0.082 – 0.266 | 0.226 | 132s | `7508a46`-dirty | threshold 0.085 |
| walnut-v0.1 | 0.195 | 0.092 – 0.306 | 0.234 | 173s | `ead1264` | vitl/sub-52k checkpoint, baseline otherwise, threshold 0.018 |

Oracle is the per-subject best threshold — the ceiling any thresholding rule could reach.

### Task 3 — brain age, fit 494 SALD / test 128 augmented DLBS

Fit on all 494 supplied SALD subjects and evaluate on the fixed augmented DLBS test with
`main_task3.py train`. DLBS is evaluation-only and must not influence submitted model weights,
calibration, or ensemble weights. The frozen walnut-v0.1 encoder was pretrained on FOMO300K, which
includes DLBS, so this tests head robustness to the imposed acquisition shifts rather than a wholly
unseen pretraining cohort.

The test set is 128 of the 464 first-wave run-1 MPRAGE scans from the public DLBS OpenNeuro
release (ds004856), selected without replacement with NumPy seed 4466, oriented to RAS and
skull-stripped with SynthSeg (matching FOMO's Task 3 preprocessing), then given one fixed-seed
composite degradation each: pose/scale, contrast/bias-field/noise, ghosting/mask erosion, and one
anisotropic-slice, isotropic-lowres, or reconstruction-blur acquisition family. Realized
per-subject parameters are embedded in the zip, and `load_fomo_task3_dlbs()` streams it like the
challenge zips. The fixed-seed augmented SALD views used by the method are cached under
`/data/smri-datasets/task3_sald_augmented` and per-view features under
`output/fomo_tune/feature_cache`, so a repeat fit with a seen checkpoint takes about a minute.

| Encoder / SALD fitting recipe | Pearson r | MAE (y) | Time | Notes |
|---|---:|---:|---:|---|
| baseline / clean only | 0.676 | 23.68 | 141s | `RidgeCV` head |
| walnut-v0.1 / clean only | 0.536 | 13.91 | 253s | `RidgeCV` head |
| walnut-v0.1 / heavy augmentation | 0.868 | 7.24 | 99s | seven views weighted to one subject |
| walnut-v0.1 / heavy augmentation + age balance | **0.878** | **7.01** | 500s | reproducible configuration below |
| baseline / heavy augmentation + age balance | 0.878 | 7.01 | 636s | augmentation closes the encoder gap |

```bash
uv run python -m fomo_tune.main_task3 train \
  name=task3_dlbs_augmented augmentation=true age_balance=true \
  ckpt_path=hf://medarc/walnut/checkpoints/walnut-v0-1/vitl/sub-52k/checkpoint-last.pth
```

Previous protocol, 20-fold over the 494 SALD subjects (kept for reference; the DLBS test above
replaces it):

| Run | Pearson r | 95% CI | MAE (y) | 95% CI | Time | Git | Notes |
|---|---|---|---|---|---|---|---|
| baseline | 0.963 | 0.957 – 0.969 | 3.69 | 3.45 – 3.95 | 306s | `1df2e5d`† | t1w, `RidgeCV` head |
| walnut-v0.1 | 0.968 | 0.963 – 0.972 | 3.50 | 3.29 – 3.74 | 261s | `ead1264` | vitl/sub-52k checkpoint, baseline otherwise |

### Task 5 — polymicrogyria, AUROC, 20-fold over 48

| Run | AUROC | 95% CI | Time | Git | Notes |
|---|---|---|---|---|---|
| baseline | 0.984 | 0.953 – 1.000 | 68s | `1df2e5d`† | t1w, `LogisticRegressionCV` |
| walnut-v0.1 | 0.995 | 0.979 – 1.000 | 69s | `ead1264` | vitl/sub-52k checkpoint, baseline otherwise |

Shas marked † predate this branch split.

### Tasks 6 and 7 — embeddings

No local metric: the challenge withholds the labels and fits its own probes. The evidence the
embedding carries signal is the three tables above, which score the same pooled vector.

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
