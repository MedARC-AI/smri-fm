---
name: synthseg-integration
description: SynthSeg fork, cost profile after the torch port (7-9x), TF32 buys nothing, and the predictor is not an nn.Module.
metadata:
  type: reference
  observed: 2026-07-29
---

Billot, Med Image Anal 2023. Fork `MedARC-AI/SynthSeg` @ `pytorch-port`. The port was written
separately beforehand and did most of the usual work.

- **Cost:** `RegularGridInterpolator` was the whole story — 3.2-3.7s/volume off-1mm, 0.5s with torch
  (7-9x). The gaussian pre-blur is a no-op for coarser-than-1mm inputs (sigmas are zeroed when
  upsampling). `rescale_volume`'s `np.percentile` (~0.24s) is now the largest remaining
  preprocessing cost. End to end ~0.74s/volume on an H100.
- **TF32 gives this U-Net nothing:** 2.7e-4 relative cost on `global_embed` for a 0.99x speedup on
  an H100. Left at the default — a model should not mutate a global backend flag — and pinned off in
  the GPU test.
- `SynthSegPredictor` is not an `nn.Module` and freezes `self.device` at construction, so the
  wrapper owns placement and never calls its `forward_embedding`.

Result: `dlbs_sex` AUROC 0.9863 (bal acc 0.947), 464 subjects in 5m44s.

See [[synthseg-numpy2-bugs]], [[synthseg-resample-length-fix]], [[synthseg-no-crop-by-default]],
[[synthseg-pooling-masks-padding]].
