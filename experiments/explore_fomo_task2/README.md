# explore_fomo_task2

Does the `fomo_tune` preprocessing keep the task-2 meningiomas? Two ways it could quietly not,
both fatal to a patch-based segmentation and neither visible in a score:

1. **The 208x240x208 fit crops the tumour off the edge.** Task 2 FOVs reach 230mm against a 208mm
   box, so ~11mm goes off each side in x — and meningiomas grow from the meninges, at the
   periphery, which is exactly where that lands.
2. **The mean-intensity brain mask excludes it.** Outside the mask the transform writes zeros, so
   an out-of-mask tumour is not merely unweighted, it is erased before the backbone sees it.

```bash
uv run python preproc_check.py     # -> preproc_check.tsv, figures/preproc_check.png
```

## Answer: neither happens

**`retained = 1.000` for all 23 subjects.** Not one labelled voxel falls outside the window the
box can see. The crop takes head, not brain — the 230mm FOVs are wide because of skull and skin.

**In-mask fraction is ~1.0**, worst cases sub-08 flair 0.922 and sub-04 dwi 0.946. Both are
small tumours where the missing few percent is a rim of boundary voxels, not a missed structure.
Median is 1.000 on both modalities.

So the coarse patch approach can be built on the transform as it stands. Neither of these is the
reason if task 2 scores badly.

## How the numbers are computed

`retained` is a **coordinate** test, not a resampling one: every op in the transform is
axis-aligned, so `inv(affine_in) @ affine_out` is diagonal and the input window the output grid
can reach is three intervals. A native foreground voxel is retained if its index is inside all
three. That avoids inventing a label resampling in order to measure one — the real pipeline will
look labels up in world coordinates, and it never resamples the seg either.

`in_mask` is measured on the output grid, over the nearest-neighbour resampled seg, against the
same `data > data.mean()` the transform applies. `fg_out` is there to read `in_mask` against: at
6mm slices upsampled to 1mm a tumour gains voxels (sub-02: 3047 -> 13002), so `fg_out` is not a
count of anything real, only the denominator.

The seg's own affine differs from the image's by up to 0.03mm. The image affine is what the
transform used and the grids match, so the seg is read on the image's affine and the difference
is ignored.

## The figure

69 rows x 10 cols, one subject per three rows (dwi, flair, flair + tumour red + brain mask cyan),
columns one acquired slice apart centred on the largest tumour cross-section. ~9600px tall, meant
for scrolling.

Two deliberate choices:

- **Panels are the full 208x240 field, uncropped.** Cropping to the brain would use the space far
  better and would hide question 1 completely.
- **One column per acquired slice, not per output slice.** Slices are 5-7mm and the transform
  resamples to 1mm, so consecutive output slices are interpolations of one acquired slice; a
  stride of 1 would have printed the same picture ten times.

Small tumours are one or two columns wide and no more. That is honest — the median tumour is
562 voxels, about 2500mm³, which is five of the backbone's 8mm patches.
