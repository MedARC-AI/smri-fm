---
name: neurojepa-fixed-input-shape
description: Neuro-JEPA must run at exactly 96x108x96 — its RoPE position decomposition entangles the y/z axes at any other shape.
metadata:
  type: project
---

Do not change Neuro-JEPA's input shape from `96x108x96`.

`RoPEAttention.separate_positions` decomposes the token index assuming `tokens_per_frame = H_p*W_p`,
but `PatchEmbed3D.flatten(2)` gives the first axis stride `W_p*D_p`. These agree only when
`D_p == W_p` — here 8 vs 9 — so at any other shape the y and z position axes are entangled. It is
self-consistent between their pretraining and their inference *at the trained shape only*.

Measured: 2x input gives pooled-feature cosine 0.926 vs native, where a 6-voxel shift control gives
0.998.

**Why:** the bug is invisible — it runs and produces plausible features at any shape.
**How to apply:** treat the shape as a hard constant in the wrapper; if a task seems to need a
different FOV, change the preprocessing, not the model input. See [[neurojepa-integration]].
