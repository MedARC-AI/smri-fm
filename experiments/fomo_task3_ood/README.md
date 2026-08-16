# Task 3 external DLBS test

Fit the current Task 3 method on all 494 challenge-provided SALD subjects and report ordinary MAE
on 128 fixed, augmented DLBS subjects. DLBS is evaluation-only; never fit or calibrate a submitted
model on it.

The external test is designed to expose reliance on resolution, reconstruction blur, apparent
anatomical scale, and other acquisition properties.

## Prepared test

On the MedARC cluster, the ready-to-use manifest and 128 images are shared at:

```text
/data/smri-datasets/task3_dlbs_external/eval.tsv
```

Each subject appears exactly once. Every image combines one randomly selected acquisition shift
(4-7 mm anisotropic slices, 2-3.2 mm isotropic acquisition, or 3-5.5 mm reconstruction blur) with
fixed-seed pose, scale, contrast, bias-field, noise, ghosting, and skull-strip variation.

## Score the walnut-v0.1 baseline

```bash
export HF_HOME="/data/smri-datasets/huggingface"
srun --partition=main --qos=high --account=sophont --gpus=1 --cpus-per-task=8 --mem=64G \
  --time=00:30:00 uv run python experiments/fomo_task3_ood/evaluate.py \
  /data/smri-datasets/task3_dlbs_external/eval.tsv
```

`evaluate.py` loads the official SALD finetuning data through `load_fomo_task3()`, fits the
unchanged walnut-v0.1 `Task3Method` on all 494 subjects, and writes `predictions.tsv` and
`score.tsv`.

| Encoder and head | Pearson r | MAE | Time |
|---|---:|---:|---:|
| walnut-v0.1 + current RidgeCV head | 0.536 | 13.91y | 250s |
| walnut-v0.1 + heavy SALD augmentation | 0.868 | 7.24y | — |
| walnut-v0.1 + heavy SALD augmentation + age-balanced fitting | **0.878** | **7.01y** | — |

Pearson r and MAE are the official Task 3 metrics and the external hill-climbing reference scores.
The two augmented fitting recipes were run locally; this PR does not include their implementation.

## Rebuild from another location

`build.py` has no cluster-specific paths. Supply an input TSV with exactly three columns:

```text
subject age path
```

The separator is a tab and `path` points to a RAS-oriented, SynthSeg-skull-stripped DLBS T1w image,
matching the preprocessing FOMO applies to Task 3 inputs. Given the full DLBS cohort, the script
deterministically selects the same 128 subjects and writes the augmented images plus `eval.tsv`:

```bash
uv run python experiments/fomo_task3_ood/build.py /path/to/dlbs_inputs.tsv /path/to/output
```
