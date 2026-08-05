---
name: neurojepa-monai-spacing-gpu
description: Neuro-JEPA's MONAI Spacing transform is 96% of preprocessing and must run on GPU — 37x, needs cupy; reordering the chain makes it slower.
metadata:
  type: project
---

`Spacing(pixdim=1mm, mode=5)` is 96% of Neuro-JEPA preprocessing: **7.03s CPU vs 0.19s GPU (37x)**
on an off-1mm volume. MONAI's bspline path needs `cupy`, hence `cupy-cuda12x` in the extra; without
it MONAI raises `OptionalImportError`.

Reordering does not help. The README's "resample first" variant measured **0.8x — slower**: the
packaged order pads or crops to 180x216x180 *before* the resample, which usually shrinks the array
first.

This only bites on off-1mm cohorts, which is most of them — ABIDE 157/250 volumes, ADHD-200
196/250. ADNI is 1mm, which is why the cliff went unnoticed until the first sweep.

**Why:** an assumption about transform ordering cost 0.8x when it was expected to win.
**How to apply:** keep the packaged order, keep the chain on GPU, and benchmark warm — cupy's first
call is ~5s of kernel JIT. See [[probe-cost-scales-with-embed-width]].
