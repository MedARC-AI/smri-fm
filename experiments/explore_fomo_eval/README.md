# Explore FOMO eval datasets

## Run

Extracted FOMO eval task data are assumed to be in `data/fomo_eval` in the project root.

```
fomo_eval/
├── FOMO26_Guide_v1.pdf
├── Task_1
│   ├── labels
│   └── preprocessed
├── Task_1.zip
├── Task_2
│   ├── labels
│   └── preprocessed
├── Task_2.zip
├── Task_3
│   ├── labels
│   └── preprocessed
├── Task_3.zip
├── Task_4
│   ├── labels
│   └── preprocessed
├── Task_4.zip
├── Task_5
│   ├── labels
│   └── preprocessed
├── Task_5_extract.py
└── Zhang_Lingfeng_2022_PPMR_Dataset.zip
```

Describe datasets:

```bash
uv run python describe.py > describe_output.txt
```

Generate plots:

```bash
# All modalities side by side, one figure per task
uv run python montage.py --batch-id {0..4}
uv run python montage.py --tasks Task_5 --batch-id {5..9}

# Per-task grid: one view/modality per task, subjects x slices
uv run python montage_grid.py --batch-id {0..4..2}
uv run python montage_grid.py --tasks Task_5 --batch-id {5..9..2}
```

## Notes

Five FOMO26 downstream eval tasks. Layout: `Task_N/{preprocessed,labels}/sub-XX/ses-01/`.

| Task | Type | n | Modalities | Target |
|---|---|---|---|---|
| 1 Infarct | cls | 21 | dwi_b1000, adc, flair, swi/t2s | 13 pos / 8 neg |
| 2 Meningioma | seg | 23 | dwi_b1000, flair, swi/t2s | mask {0,1} |
| 3 Brain age | reg | 494 | t1w | age 19–80 |
| 4 Trigeminal | seg | 40 | t2w | mask {0,1,2} |
| 5 Polymicrogyria | cls | 48 | t1 | 24 / 24 |

### What the targets are

- **Task 1 — Infarct:** is there an acute stroke (dead tissue from blocked blood flow)? Per-subject yes/no. Positives also carry a lesion mask.
- **Task 2 — Meningioma:** segment the tumour. A usually-benign mass growing from the brain's outer lining, not the brain itself.
- **Task 3 — Brain age:** predict the subject's age in years from a healthy T1 scan. Gap between predicted and true age is used as a brain-health marker.
- **Task 4 — Trigeminal neuralgia:** segment the trigeminal nerve and the blood vessel touching it. Vessel-on-nerve compression causes severe facial pain; label 1 vs 2 are the two structures, present on both sides.
- **Task 5 — Polymicrogyria:** does the cortex have too many small folds (a malformation from abnormal brain development)? Per-subject yes/no.

### Watch out

- **"preprocessed" ≈ reorient (RAS) + within-subject coreg only.** No common resolution, no intensity norm, inconsistent skull-strip (stripped: 1, 3; full head: 2, 4, 5). Intensities span orders of magnitude. We normalize ourselves.
- **Tasks 1–2 are thick-slice** (5–7 mm, ~21–30 slices, sub-mm in-plane). Task 3 is 1 mm iso, Task 4 ~0.5 mm near-iso, Task 5 anisotropic.
- **Seg targets are tiny**: lesion volume spans ~3–4 orders of magnitude (Task 1/2: 12→90k voxels; Task 4: 266→2277). Mean Dice will be unstable — report per-subject and stratify by size.
- **No official splits, n is small.** Test folds of 2–5 subjects. Use k-fold × multi-seed and report spread, not a single number.

### Design decisions from this review

- **Task 1: use dwi_b1000 ∧ adc, not dwi-only.** Infarct = restricted diffusion (DWI bright *and* ADC dark); confirmed in 10/13 positives. DWI alone confuses T2 shine-through — likely the "false negatives" seen in the images are correctly-unlabeled shine-through.
- **Task 2 lacks contrast T1** (the standard meningioma sequence). Genuinely hard on this subset; expect weak scores.
- **Task 4 wants 3D.** Nerve is a thin A-P tract; no single 2D plane captures it. Spacing is near-iso, so 3D patches fit. Labels 1/2 are two adjacent structures (nerve vs. vessel), both bilateral — not left/right.
- **Task 5 is questionable.** Reconstructed from lossy JPEG slices stacked along a coarse (1.2 mm) A-P axis; ringing + per-subject spacing hacks. Target is small cortical folds (polymicrogyria), which could be blurred.

### Open threads

- Task 3 age distribution is bimodal → possible two-scanner confound a model could cheat on. Unchecked.
- Cross-task duplicate-subject check. Unchecked.
