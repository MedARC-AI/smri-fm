# explore_skull_stripping

The MAE was pretrained on SynthSeg-stripped volumes. `SmriMaeTransform` masks with
`data > data.mean()` instead, which keeps skull, scalp and neck. That gap has been a known,
unmeasured assumption since July.

This measures it and asks whether closing it with SynthSeg is the right move **per task**, because
the tasks do not want the same thing: two of them ship volumes that are already stripped, and two
of them label structures at or outside the brain boundary, where a brain mask is a way to delete
the answer.

```bash
# one shard per task, ~25s/volume, ~20GB of GPU each -- three at a time fits one H100
for t in task1 task2 task3 task4 task5; do
    uv run python survey.py --device cuda --tasks $t --n-subjects 48 > /dev/null &
done
uv run python -c "import glob, pandas as pd; \
    pd.concat(map(lambda f: pd.read_csv(f, sep='\t'), sorted(glob.glob('survey_task*.tsv')))) \
      .to_csv('survey.tsv', sep='\t', index=False, float_format='%.4f')"

uv run python figure.py --device cuda        # -> figures/masks.png
uv run python pretrain_mask.py --device cuda # -> pretrain_mask.tsv
```

## What pretraining actually did

From `/data/connor/smri-fm/src/preprocessing/pipeline.py`, plus the one FOMO300 subset copied
whole into `data/PT001_ClevelandCCF/`:

1. ANTs **rigid** registration of the native scan to TemplateFlow `MNI152NLin2009cAsym`, 1mm,
   bSpline → `processed/*_desc-processed.nii.gz` at 193x229x193. Not stripped: 74% nonzero.
2. SynthSeg with `--parc --robust` run **on the registered image**, same grid.
3. `mask = seg > 0`, no morphology whatsoever (`save_brain_mask_from_segmentation`).
4. `build_fomo300k_sparse_wds.py`: centre fit to 208x240x208, z-score over masked voxels *after*
   the fit, store fp16 brain voxels plus a bit-packed mask. Everything outside is exactly 0 and is
   never a live token.

So the eval transform differs from pretraining in **two** ways, not one: no brain mask, and no
rigid registration. Registration is deliberately out of scope here; it is the larger remaining gap.

An oddity, noted and set aside: for `PT001 sub-01` the stored mask and the stored seg disagree
(Dice 0.973 — the mask drops 52k of the 208k label-24 voxels and adds 31k where `seg == 0`), which
the pipeline code says cannot happen. Most likely the seg was regenerated after the mask was
written. `pretrain_mask.tsv` carries this per subject as `dice_stored_reference`.

## Where SynthSeg runs

On the grid the transform already produces. SynthSeg pads to a multiple of 32 internally and crops
back, so feeding it a volume that is already RAS 1mm 208x240x208 returns the segmentation **on that
same grid** — no mask resampling, no second interpolation. That is the whole integration point.

`third_party/SynthSeg`, branch `pytorch-port`. The port implements only the default single-network
path; `--robust` would mean the 215MB checkpoint plus porting a denoiser and a second network, and
`--parc` is irrelevant to a `seg > 0` mask. `pretrain_mask.py` measures what that costs.

## Answer: strip task 5, leave the rest alone

The gap is real and large — but SynthSeg is the right fix for exactly one task, and for task 2 it
is a catastrophe. Coverage is every subject of tasks 1, 2, 4 and 5; task 3 is 40 of 494.

### Live tokens, median of 20280

| task | modality | `thr` today | `pos` | `brain` |
|---|---|---|---|---|
| task1 | adc | 3577 | 3635 | 3317 |
| task1 | dwi_b1000 | 3502 | 3635 | 3249 |
| task1 | flair | 3521 | 3635 | 3373 |
| task2 | dwi_b1000 | 5421 | 13711 | 3017 |
| task2 | flair | 6595 | 13855 | 3268 |
| task3 | t1w | 3427 | 3431 | 3467 |
| task4 | t2w | 7923 | 13973 | 3414 |
| task5 | t1 | 8130 | 11380 | 3946 |

Tasks 1 and 3 are already stripped and all three masks agree. Tasks 2, 4 and 5 pool roughly
**twice as many tokens as there is brain**, and `pos` is worse than useless there — those volumes
carry a low-level background that `data > 0` promotes to live tokens, and one voxel above the
threshold validates a whole 8mm patch.

### Fraction of the task's own label the mask keeps

| task | modality | `thr` med / min | `pos` med / min | `brain` med / min |
|---|---|---|---|---|
| task1 | adc | 1.000 / 0.970 | 1.000 / 0.976 | 1.000 / **0.933** |
| task1 | dwi_b1000 | 1.000 / 0.971 | 1.000 / 0.976 | 0.989 / **0.030** |
| task1 | flair | 1.000 / 0.969 | 1.000 / 0.976 | 1.000 / **0.930** |
| task2 | dwi_b1000 | 1.000 / 0.946 | 1.000 / 1.000 | **0.467** / 0.000 |
| task2 | flair | 1.000 / 0.922 | 1.000 / 1.000 | **0.868** / 0.000 |
| task4 | t2w | 1.000 / 0.974 | 1.000 / 1.000 | 1.000 / **0.872** |

This is what settles it, and it settles it against stripping in three of the four labelled cases:

- **Task 2 is disqualifying.** The median meningioma loses **half its voxels** on DWI, and the
  minimum is **0.000** — tumours erased outright. Meningiomas are dural, so they grow exactly
  where the brain boundary runs. Outside the mask the transform writes zeros, so this is deletion,
  not down-weighting.
- **Task 1 on DWI is dangerous.** One subject retains **3%** of its infarct. SynthSeg also drops a
  median 16% (max 53%) of the already-stripped support on DWI, so it is failing on the contrast,
  not tightening a boundary. FLAIR and ADC are fine — but tasks 1 and 3 need nothing anyway.
- **Task 4 mostly survives** — median 1.000, worst case 0.872. The cisternal trigeminal segment
  sits inside SynthSeg's CSF label. Not disqualifying, but it costs ~13% of the target on the
  worst subject against a 2.3x token gain, and the task is not started.
- **Task 5 is the clean case**: no voxelwise target to protect, 8130 tokens down to 3946.

### What SynthSeg costs against `data > 0`, where the volume is already stripped

| task | modality | brain outside `pos` med/max | `pos` dropped med/max |
|---|---|---|---|
| task1 | dwi_b1000 | 0.001 / 0.004 | 0.158 / 0.532 |
| task1 | flair | 0.002 / 0.006 | 0.072 / 0.240 |
| task3 | t1w | 0.014 / 0.021 | 0.004 / 0.008 |

On task 3 the two masks are interchangeable. On task 1 SynthSeg only removes, and what it removes
is sometimes the lesion.

### Port fidelity (`pretrain_mask.tsv`, 8 PT001 subjects)

Default single-network path against the pipeline's own `--parc --robust` mask, same grid:
**Dice 0.963**, the port running **6.4% larger** and missing 1.2%. For reference the stored seg
disagrees with the stored mask by Dice 0.970 on the same subjects, so the port is about as close
to the pretraining mask as the pretraining pipeline's own artifacts are to each other. Good enough
— an order of magnitude below the 2x token error being fixed.

## Proposal

Not one mask for every task — the cheapest mask that is correct for each. Nothing here argues for
a uniform pipeline, and three of the five tasks are better off without SynthSeg entirely.

| task | mask | why |
|---|---|---|
| 1 infarct | `data > 0` | already stripped; SynthSeg keeps 3% of one subject's lesion on DWI |
| 2 meningioma | `data > data.mean()`, unchanged | a brain mask deletes the median tumour's other half |
| 3 brain age | `data > 0` | already stripped; interchangeable with SynthSeg, and free |
| 4 trigeminal | undecided | survives at median 1.000 / min 0.872 against a 2.3x token gain |
| 5 polymicrogyria | SynthSeg | no voxelwise target, 8130 tokens down to 3946 |

`data > 0` for tasks 1 and 3 is not a compromise. It is strictly safer than the mean threshold
that ships today — same footprint on those cohorts, no dependency, and no failure mode — so it is
worth doing even though its effect on the score will be near zero.

Consequences worth stating before any of it is written:

- **Only task 5 pays for packaging.** A 53MB checkpoint pre-converted to a torch `state_dict` at
  build time (so `h5py` drops out of the container), plus `scipy`. The container has a GPU and a
  900s per-subject budget against a few seconds of inference, so runtime is not a constraint.
- **Task 5's committed AUROC 0.984 will move**, and that is the point — it is the cohort where
  more than half the pooled tokens are currently skull and neck. Tasks 1 and 3 should not move.
- **`backbone.py` stops being one transform for every task.** Whatever shape that takes, the mask
  becomes a per-task choice rather than a property of the backbone.
- Registration remains unaddressed and is the larger remaining gap.

## What this does not settle

A **dilated** brain mask is the obvious way to keep task 2 and task 4 in play: enough margin to
spare a dural tumour while still excluding scalp and neck. The sweep is cheap — one segmentation
per volume, retention and token count re-measured at each radius — and it is the next thing to run
if tasks 2 or 4 want the fidelity. Nothing here measures it.

Nor does anything here measure whether a correct mask *improves a score*. It closes a known gap
between eval and pretraining; that is an argument from fidelity, not evidence.
