# explore_fomo_task5

Task 5 asks whether a T1 shows polymicrogyria, and it has always scored suspiciously well: AUROC
0.984 in `fomo_tune_baseline`, 0.995 in `fomo_tune_walnut_v0_1`, both out of fold at n=48. This
looks at the images, and asks what the score is made of.

```bash
uv run python experiments/explore_fomo_task5/segment.py --device cuda \
    | tee experiments/explore_fomo_task5/output/segment.log     # 48 SynthSegs, ~25 min on one GPU
uv run python experiments/explore_fomo_task5/cortex.py          # -> figures/cortex_*.png
uv run python experiments/explore_fomo_task5/slices.py          # -> figures/clipping.png
for y in 0 1; do uv run python experiments/explore_fomo_task5/montage.py \
    --label $y --plane coronal --native; done                   # -> figures/*_coronal_native.jpg
uv run python experiments/explore_fomo_task5/strip.py           # -> output/*/t1_1mm_brain.nii.gz,
                                                                #    tight seg > 0, CSF class dropped
uv run python experiments/explore_fomo_task5/explore.py \
    | tee experiments/explore_fomo_task5/output/explore.log     # -> explore.tsv, figures/scalars.png
```

`output/*.nii.gz` is gitignored: 1.4GB of resampled volumes, brain-masked copies and label
maps, all regenerable from `segment.py` and `strip.py`.

## The answer: the controls are cut through the brain and the cases are not

Every one of the 24 controls has its anterior-posterior field of view clipped inside the
cerebrum.

One scalar, the fraction of the peak brain cross-section left in the outermost AP slice:

| | controls | cases |
|---|---|---|
| brain in the edge slice | 0.229 ± 0.055 | 0.024 ± 0.043 |
| above 0.10 | **24 / 24** | **2 / 24** |
| AP margin to the FOV edge | 0mm, all 24 | 11.9mm mean |
| AP field of view | 151mm (133–167) | 199mm (146–250) |

That single scalar scores **AUROC 0.997** with a permutation p below 1e-4, against the model's
0.984 and 0.995.

It needs no measurement to see. `figures/clipping.png` draws one axial slice per subject at the
full field of view, brain outline in cyan and the part of it lying against the FOV edge in red.
Every control panel has a red bar across the top and the bottom. Most case panels have air above
and below the head. `figures/cortex_l_lateral.png` shows the same thing on the surface: the
control brains end in flat vertical walls front and back, and the cases are rounded.

The grouped baselines say the same thing. Leave-one-out logistic heads on scalars that need no
backbone:

| features | LOO AUROC |
|---|---|
| SynthSeg tissue volumes | 0.759 |
| nifti header alone (FOV, voxel size, matrix) | 0.896 |
| **coverage (edge fraction, margin, brain volume)** | **0.995** |
| all 12 scalars | 0.991 |
| the frozen MAE + logistic head | 0.984 / 0.995 |

And the model is reading it. Rank-residualize the out-of-fold probability against the edge
fraction and the score falls from 0.995 to **0.632**, spearman −0.809. For comparison, the same
operation on task 1 against its head-size confound left 0.865. Task 5's confound accounts for
almost all of the ranking.

## Where it comes from

`data/fomo_eval/Task_5_extract.py` builds these volumes by stacking coronal JPEGs exported from
PACS, from the Zhang 2022 PPMR thesis dataset. Cases and controls came out of that archive as
different kinds of export. The case folders carry a full or near-full coronal stack; the control
folders carry a clinically cropped one.

Scanner is not the explanation. The split is balanced across classes, and neither the in-plane
matrix (AUROC 0.479) nor the AP voxel size (0.482) separates anything:

| | controls | cases |
|---|---|---|
| 3T Skyra | 10 | 11 |
| 1.5T Cigna GE | 14 | 13 |

`figures/clipping.png` colours each panel title by scanner, and the clipping tracks the class
inside both groups. Every Skyra control is clipped and no Skyra case is; the same holds for
Cigna. It is coverage, not site.

*(CL)* However, there is another confound that is specific to the Skyra site. All Skyra
cases have static white noise bands at the anterior and posterior ends, whereas Skyra
controls do not (because all controls, Skyra as well as Cigna, are clipped). Meanwhile,
none of the Cigna cases or controls have these noise bands.  This could explain why a
model trained on Skyra could fail to generalize to Cigna, but not vice versa.

### How the scanner is known

It is read back off the in-plane matrix of the nifti: 260x320 is the Skyra, 512x512 the Cigna.
That works because `Task_5_extract.py` resizes every JPEG slice to the chosen scanner's native
reconstructed matrix, so the shape on disk records the choice exactly, and there are only two
values across all 48 subjects.

## Per-scalar table

| scalar | AUROC | perm p | control / case |
|---|---|---|---|
| brain in edge slice | 0.003 (0.997 inverted) | <1e-4 | 0.229 / 0.024 |
| AP field of view, mm | 0.924 | <1e-4 | 151 / 199 |
| AP margin, mm | 0.812 | <1e-4 | 0 / 11.9 |
| slice count | 0.842 | 1e-4 | 142 / 188 |
| ventricle volume, mL | 0.760 | 0.0012 | 15.7 / 31.7 |
| gm/wm ratio | 0.545 | 0.63 | 1.35 / 1.34 |
| cortical folding index | 0.382 | 0.16 | 149 / 146 |
| cortex volume, mL | 0.405 | 0.26 | 608 / 578 |
| cortex fraction of brain | 0.523 | 0.80 | 0.376 / 0.374 |
| brain volume, mL | 0.377 | 0.14 | 1620 / 1550 |
| in-plane matrix | 0.479 | 0.79 | — |
| AP voxel size | 0.482 | 0.78 | — |

Two of these are worth separating from the confound.

**Ventricular enlargement is probably real.** Cases carry twice the ventricular volume, p=0.0012,
and ventriculomegaly is a documented associate of polymicrogyria. It is also the one anatomical
scalar that survives, so a model that read only anatomy still has something to work with.

**Cortical folding does not separate at all** (0.382, p=0.16), which is the surprise given that
polymicrogyria is by definition excess folding. The index is the 6-neighbour boundary area of
SynthSeg's cortex label over its volume to the two-thirds power. The most likely reading is that
SynthSeg at 1mm does not resolve microgyri, so the label describes a smoothed cortex; the renders
support that, since the case surfaces do not look obviously more folded than the controls. This
is a limit of the measurement, not evidence that the cases lack the pathology.

`montage.py` was written to check that by going back to the intensities. It did not settle it:
see below, where the attempt to read the cortex by eye failed.

## Reading the cortex directly

The scans are acquired **coronal**, at 0.43–0.77mm in plane against 0.7–1.2mm through plane, so
coronal is the only orientation with sub-millimetre spacing in both in-plane axes. Axial and
sagittal are reformats across the slice direction and lose most of the detail. `--plane` takes
any of the three; `--native` slices the original nifti rather than SynthSeg's 1mm resampling,
carrying the brain mask over by affine, which is worth roughly 2x in plane on the Cigna subjects.
Panels are masked with a 3mm-dilated brain mask, which is what removes the noise bands some
exports carry outside the head.

Comparing `figures/pmg_coronal_native.jpg` against `figures/control_coronal_native.jpg`,
we tried to read the cortex by eye. *(CL)* But overall, it's hard to visually tell the cases
from controls (neuroradiology is not trivial I guess).

*(CL)* In a few PMG cases, I noticed some subtle visual differences. E.g. for Sub 45,
bumpy cortex around mid-lateral central sulcus bilaterally.

## Figures

- `clipping.png` — all 48 axial slices at the widest part of the brain, drawn at each subject's
  own field of view with no cropping. Cyan is the brain outline, red is where it runs into the
  edge of the scan, and the title colour is the scanner. This is the whole result in one picture.
- `cortex_l_lateral.png`, `cortex_r_lateral.png` — all 48 cortical surfaces, one hemisphere per
  figure, controls in the first three rows and cases in the last three. The clipping is the
  thing to look at.
- `cortex_views.png` — four controls and four cases at lateral and medial, both hemispheres.
- `pmg_coronal_native.jpg`, `control_coronal_native.jpg` — 24 subjects by 8 coronal slices 8mm
  apart, centred on each subject's widest slice, on the acquired grid, masked and cropped to
  the brain. 1536x4608, so open them at full size and zoom.
- `sub_45_screenshot.png` — sub_45 in niivue, on the brain-masked native grid.
- `scalars.png` — the model's out-of-fold probability against the edge fraction, the AP field of
  view, the folding index and ventricular volume.

The surfaces are rendered without a mesh library: rays are cast along x through SynthSeg's cortex
label, and the depth of the first hit is shaded by its own gradient plus a cavity term. Nothing
is smoothed or decimated, so what the render loses is what the 1mm segmentation lost.

A rendering caveat before reading pathology off these: surface texture tracks the scanner. The
Skyra subjects (260x320 in plane) come out grainier than the Cigna ones (512x512) because
SynthSeg is resampling from a coarser grid. That is acquisition, not anatomy.

## What this suggests next

1. **The score is not evidence that the backbone detects polymicrogyria**, and it should not be
   read as such in the leaderboard table.
2. **Crop every subject to a common AP extent** and re-run the baseline. The controls set the
   budget at roughly 133mm, so cropping every case to its brain's central 133mm removes the cue
   without touching the cases' anatomy.
