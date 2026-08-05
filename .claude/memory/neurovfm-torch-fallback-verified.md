---
name: neurovfm-torch-fallback-verified
description: NeuroVFM's pure-torch fallback was diffed against flash-attn end to end at fp32 rel err 1.2e-6 — it is exact, not an approximation, but the layer_norm residual is subtle.
metadata:
  type: project
---

With flash-attn temporarily installed on an H100, diffed end to end through the encoder:
**fp32 rel err 1.2e-6**, bf16 6.6e-3 (bf16 eps is 7.8e-3). `FusedDense` vs `F.linear` and `FusedMLP`
vs `fc2(gelu(fc1, approximate="tanh"))` are bit-exact.

The one part that is easy to get wrong: `layer_norm_fn(prenorm, residual_in_fp32)` accumulates the
residual in fp32 and normalizes *that*, so a naive `x + residual` in the input dtype drifts.
`neurovfm/models/torch_fallback.py` reproduces it.

**Why:** without the fp32 diff there would be no evidence the fallback is faithful, and bf16
rounding (~7e-3) would have masked a real algorithmic difference.
**How to apply:** treat the fallback as the reference path. See [[flash-attn-rejected]].
