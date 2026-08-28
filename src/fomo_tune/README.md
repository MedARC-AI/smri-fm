# fomo_tune

The FOMO26 challenge tasks, one script each, tuned independently.

## Layout

| File | |
|---|---|
| `main_task<k>.py` | One task, end to end. Each has a **frozen** "protocol" section. Anything outside of the protocol is fair game. |
| `datasets.py` | **frozen**. One `load_fomo_task<k>()` per task, streaming the challenge zips into an HF dataset. Raw niftis, no resampling — the backbone transform does that. |
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

| Task | n | Inputs | Output | Split |
|---|---|---|---|---|
| 1 infarct | 21 | adc, dwi_b1000, flair (+t2s/swi) | probability | LOO |
| 2 meningioma | 23 | dwi_b1000, flair (+t2s/swi) | mask, input grid | LOO |
| 3 brain age | 494 | t1w | age in years | 20-fold |
| 4 trigeminal | 40 | t2w | mask, labels 1=nerve 2=vessel | LOO |
| 5 polymicrogyria | 48 | t1w | probability | 20-fold |
| 6+7 probing, fairness | — | one image, any modality | 1024-d embedding `.npy` | — |

## Leaderboard

### Task 1 — infarct, AUROC, LOO over 21

| Run | AUROC | 95% CI | Time | Git | Notes |
|---|---|---|---|---|---|
| baseline | 0.990 | 0.944 – 1.000 | 11s | `1df2e5d`† | dwi_b1000 only, `LogisticRegressionCV` |
| walnut-v0.1 | 0.894 | 0.731 – 1.000 | 11s | `ead1264` | vitl/sub-52k checkpoint, baseline otherwise |
| walnut-v0.1 ensemble | 0.990 | 0.942 – 1.000 | 38s | `5e078be` | zero masking, ensemble pooling, volume-normalized — the current default |

### Task 2 — meningioma, Dice, LOO over 23

| Run | Dice | 95% CI | Oracle | Time | Git | Notes |
|---|---|---|---|---|---|---|
| baseline | 0.195 | 0.098 – 0.303 | 0.271 | 174s | `7d13f45` | flair only, largest-component filter, threshold 0.011 |
| no largest component | 0.170 | 0.082 – 0.266 | 0.226 | 132s | `7508a46`-dirty | threshold 0.085 |
| walnut-v0.1 | 0.195 | 0.092 – 0.306 | 0.234 | 173s | `ead1264` | vitl/sub-52k checkpoint, baseline otherwise, threshold 0.018 |
| progressive grid: walnut-v0.1 | 0.277 | 0.149 – 0.408 | 0.345 | 1022s | `27e43b2` | 9 progressive CNN heads, uniform average, threshold 0.401 — the current default |
| progressive grid: pretrain_full | 0.286 | 0.159 – 0.418 | 0.326 | 1062s | `27e43b2` | otherwise identical, threshold 0.257. Ties walnut |

Oracle is the per-subject best threshold — the ceiling any thresholding rule could reach.

### Task 3 — brain age, 20-fold over 494

| Run | Pearson r | 95% CI | MAE (y) | 95% CI | Time | Git | Notes |
|---|---|---|---|---|---|---|---|
| baseline | 0.963 | 0.957 – 0.969 | 3.69 | 3.45 – 3.95 | 306s | `1df2e5d`† | t1w, `RidgeCV` head |
| walnut-v0.1 | 0.968 | 0.963 – 0.972 | 3.50 | 3.29 – 3.74 | 261s | `ead1264` | vitl/sub-52k checkpoint, baseline otherwise |
| zero masking | 0.969 | 0.964 – 0.973 | 3.45 | 3.23 – 3.68 | 4s | `a54f1d4` | + `data > 0` token mask, final block, one view per subject |
| train views | 0.965 | 0.959 – 0.970 | 3.60 | 3.36 – 3.86 | 7s | `a54f1d4` | + fit on corrupted views of each subject — the current default |

**CamCAN OOD transfer**

| Run | Pearson r | 95% CI | MAE (y) | 95% CI | Time | Git | Notes |
|---|---|---|---|---|---|---|---|
| baseline | 0.453 | 0.385 – 0.516 | 29.33 | 27.89 – 30.86 | 431s | `a66ed36`-dirty | head fit on all 494 |
| + SynthSeg strip | 0.947 | — | 5.43 | — | — | `a54f1d4` | the gap was a skull-strip mismatch |

**CamCAN under corrupted input**, r / MAE, head fit on all 494. `experiments/task3_perturb`.

| Train views | clean | thick slice 5mm | acquired at 2mm | random scale |
|---|---|---|---|---|
| clean only | 0.947 / **5.43** | 0.919 / 9.32 | 0.928 / 7.90 | 0.884 / 8.13 |
| + resolution | 0.942 / 6.14 | 0.942 / **5.66** | 0.941 / **6.16** | 0.908 / 7.47 |
| + resolution + scale | 0.946 / 5.99 | 0.945 / 5.67 | 0.943 / 6.54 | 0.945 / **6.04** |

### Task 4 — trigeminal, mean Dice over the two labels, LOO over 40

| Run | Dice | 95% CI | Oracle | Time | Git | Notes |
|---|---|---|---|---|---|---|
| logistic: walnut-v0.1 | 0.355 | 0.302 – 0.402 | 0.385 | 1091s | `2eb1685` | alpha 1e1, cut 0.157 on both labels — the current default |
| logistic: pretrain_full | 0.339 | 0.283 – 0.387 | 0.370 | 1093s | `2eb1685` | otherwise identical |
| ridge: walnut-v0.1 | 0.274 | 0.228 – 0.320 | 0.300 | 842s | `2eb1685` | the previous head, on the better checkpoint |
| ridge: pretrain_full | 0.256 | 0.211 – 0.301 | 0.283 | 871s | `2eb1685` | the baseline below, re-measured on the truncated cut grid |
| baseline | 0.252 | 0.212 – 0.293 | 0.275 | 1216s | `51772f4` | ridge, scale 4, subcell 4, depth 4 |
| scale 3 | 0.199 | 0.162 – 0.234 | 0.219 | 762s | `51772f4` | otherwise identical |
| scale 2 | 0.130 | 0.106 – 0.156 | 0.148 | 786s | `1889e8b` | otherwise identical |
| first sweep | 0.082 | 0.068 – 0.097 | 0.093 | 1956s | `69c2d36` | one shared cut, final block |

The head is worth +0.082 and the checkpoint +0.017, each reproducing under the other.
`experiments/task4_logistic`. Rows above the baseline use a threshold grid truncated to 1e-3, worth
about +0.004 on its own.

*Nb, NSD metric is not yet implemented.*

### Task 5 — polymicrogyria, AUROC, 20-fold over 48

| Run | AUROC | 95% CI | Time | Git | Notes |
|---|---|---|---|---|---|
| baseline | 0.984 | 0.953 – 1.000 | 68s | `1df2e5d`† | t1w, `LogisticRegressionCV` |
| walnut-v0.1 | 0.995 | 0.979 – 1.000 | 69s | `ead1264` | vitl/sub-52k checkpoint, baseline otherwise |
| synthseg + AP crop | 0.882 | 0.774 – 0.972 | 127s | `c9c8f6a` | vitl/sub-52k, stripped and cropped to a common 133mm AP slab — the current default |

Shas marked † predate this branch split.

### Tasks 6 and 7 — embeddings

No local metric: the challenge withholds the labels and fits its own probes. The evidence the
embedding carries signal is the three tables above, which score the same pooled vector.
