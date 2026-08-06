---
name: synthseg-resample-length-fix
description: SynthSeg's resample is exactly F.interpolate(align_corners=False), except their output length is ceil(n*f) where torch gives floor — fix by padding the source, not the output.
metadata:
  type: project
  observed: 2026-07-29
---

Their grid `start=-(f-1)/(2f), step=1/f` simplifies to `(i+0.5)/f - 0.5`, which is exactly
`align_corners=False`. Verified at 1e-12 vs scipy. Do not hand-roll it.

One gap: their length is `ceil(n*f)` where torch gives `floor`, and their extra sample is
edge-clamped. Fix by replicate-padding the **source** by `ceil(1/f)` and trimming to `ceil(n*f)`.
Padding the *output* is wrong — the appended value is the source edge voxel, not the last output
slice. Agreement after the fix: 6.7e-6 of range.

This one voxel matters because of the uncropped padding cliff: one voxel of FOV difference flips the
pad target by 32 (193 -> 224 vs 192 -> 192), which moved `global_embed` cosine to 0.998. After the
source-pad fix, 1.00000000.

**Why:** an off-by-one in output length looks harmless and turned into a large feature change
through a downstream padding threshold.
**How to apply:** when matching a reference resampler, check the output *length* convention
separately from the sampling grid. See [[synthseg-integration]].
