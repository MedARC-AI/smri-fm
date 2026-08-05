---
name: synthseg-pooling-masks-padding
description: SynthSeg scans occupy 24-99% of the padded volume, so an unmasked mean pool mixes brain and padding in a subject-dependent ratio; _bottleneck_box ceils the start and floors the stop.
metadata:
  type: project
---

Because the default path does not crop, the scan occupies anywhere from 24% to 99% of the padded
volume depending on FOV. An unmasked mean pool therefore mixes brain and padding in a ratio that
varies per subject — a subject-dependent bias, not a constant offset.

`_bottleneck_box` keeps cells strictly inside the scan: **ceil** for the start, floor for the stop.
Flooring both included ~14% pure padding.

Still approximate — a cell's receptive field reaches past the STRIDE voxels it covers.

**Why:** the padding fraction correlating with FOV means it can correlate with site or scanner,
which is exactly the confound the benchmark is trying to avoid.
**How to apply:** any new pooled-feature backbone that pads should get the same treatment. See
[[synthseg-no-crop-by-default]].
