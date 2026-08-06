---
name: neurovfm-integration-notes
description: NeuroVFM fork, public weights and arch, plus two wrapper gotchas — SimpleITK input conversion and a hardcoded bfloat16 in SelfAttention.
metadata:
  type: reference
  observed: 2026-07-29
---

MLNeurosurg, Nat Med 2026. Fork `clane9/neurovfm`. Weights `mlinslab/neurovfm-encoder` are public:
ViT-B, patch 4x16x16 at 1x1x4mm, varlen tokens + coords, background tokens dropped, ~85.8M params.

- `preprocess_image()` takes a SimpleITK image, so the wrapper converts the nifti in memory
  (RAS affine -> LPS origin/direction) rather than writing a temp file.
- `SelfAttention` hardcodes `dtype=torch.bfloat16` on its linears, so a pure-fp32 forward needs
  `model.float()` first.

See [[neurovfm-dependency-traps]], [[neurovfm-torch-fallback-verified]],
[[sdpa-not-faster-at-real-token-counts]].
