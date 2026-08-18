# Task 3 external DLBS development test

The Task 3 protocol fits only on the 494 challenge-provided SALD subjects, then reports Pearson r
and MAE on 128 fixed, augmented Dallas Lifespan Brain Study (DLBS) subjects. Never fit or calibrate
a submitted model on DLBS.

The frozen walnut-v0.1 encoder was pretrained on FOMO300K, which includes DLBS. This is therefore a
test of head robustness to the imposed acquisition/domain shifts, not a wholly unseen pretraining
cohort.

## Evaluation cohort

The source data are the first-wave, run-1 MPRAGE scans and MRI ages from the public
[DLBS OpenNeuro release](https://openneuro.org/datasets/ds004856/versions/1.3.0). We sorted the 464
available subjects and selected 128 without replacement using NumPy seed 4466. Before augmentation,
each scan was oriented to RAS and skull-stripped with SynthSeg, matching the preprocessing FOMO
applies to Task 3 inputs.

Each subject contributes exactly one fixed composite image. Every image receives all of the
following:

- rotation up to 11.2 degrees per axis, translation up to approximately 11.2 mm, and apparent
  isotropic scale from 0.86 to 1.14;
- gamma contrast from 0.615 to 1.495, a smooth linear bias field, and magnitude noise with sigma
  2.75% of the 1st-to-99th-percentile brain intensity range;
- one-axis ghosting shifted by 6-12 mm with mixing weight 0.18-0.28, plus 1-3 mm of brain-mask
  erosion;
- one randomly selected acquisition family: 4-7 mm anisotropic boxcar slices, 2-3.2 mm isotropic
  Gaussian acquisition, or 3-5.5 mm FWHM reconstruction blur.

The random seed and realized parameters are fixed. `load_fomo_task3_dlbs()` in `datasets.py`
downloads the prepared zip and exposes the same `subject`, `age`, and `t1w` fields as the SALD
loader.

## Reproduce the augmented fit

`main_task3.py` creates six deterministic SALD views in addition to each clean image: moderate
acquisition degradation, extreme low resolution, geometry/scale, intensity/artifact,
motion/coverage, and a composite domain shift. Their per-subject fitting weights are respectively
0.15, 0.10, 0.15, 0.15, 0.10, and 0.10; the clean view receives 0.25, so every SALD subject has
total weight one.

The age-balanced recipe first selects the ridge alpha using clean SALD only. It then fits a ridge
head to all seven views using inverse-frequency weights for six SALD age bins (18-29, 30-39,
40-49, 50-59, 60-69, and 70-80), in addition to the view weights. DLBS labels are used only after
fitting to compute the two metrics.

```bash
export HF_HOME="/data/smri-datasets/huggingface"
srun --partition=main --qos=high --account=sophont --gpus=1 --cpus-per-task=8 --mem=64G \
  --time=01:00:00 uv run python -m fomo_tune.main_task3 train \
  name=task3_dlbs_augmented augmentation=true age_balance=true \
  ckpt_path=hf://medarc/walnut/checkpoints/walnut-v0-1/vitl/sub-52k/checkpoint-last.pth
```

The command writes the fitted SALD-only model, per-subject predictions, and `metrics.json` under
`output/fomo_tune/task3_dlbs_augmented`.

| Encoder / SALD fitting recipe | Pearson r | MAE | Time |
|---|---:|---:|---:|
| current default / clean only | 0.676 | 23.68y | 141s |
| walnut-v0.1 / clean only | 0.536 | 13.91y | 253s |
| walnut-v0.1 / heavy augmentation | 0.868 | 7.24y | — |
| walnut-v0.1 / heavy augmentation + age balance | **0.878** | **7.01y** | — |

## Rebuild the prepared zip

`build.py` takes a tab-separated manifest with exactly `subject`, `age`, and `path` columns. `path`
must point to a RAS-oriented, SynthSeg-skull-stripped DLBS T1w image. Given all 464 source rows,
the script deterministically writes the 128 images, `eval.tsv`, and the standard
`Task_3_DLBS.zip` consumed by `datasets.py`:

```bash
uv run python experiments/fomo_task3_ood/build.py /path/to/dlbs_inputs.tsv /path/to/output
```
