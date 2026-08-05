---
name: eval-interface-is-nifti-in
description: The eval suite uses a uniform per-nifti model interface so segmentation scoring is independent of any model's patch grid — the accepted cost is that batched inference is gone.
metadata:
  type: project
---

`src/nanobrain/eval` moved from a `(model, transform)` contract to a uniform per-nifti interface:
`global_embed(nifti) -> (D,)` and `dense_embed(nifti) -> (X, Y, Z, D)` on the RAS-canonical grid.
Models canonicalize, normalize and own device placement internally. Shared `canonical()` in
`nifti.py` is the grid contract the seg probe aligns labels to. Commit `50cb2a8`.

Motivation: make segmentation scoring independent of any model's patch grid, so backbones are
comparable. **Accepted cost: batched inference is gone** — one volume at a time.

**Why:** the single-volume throughput limit is a deliberate trade for comparability, so it will keep
looking like an easy optimization to someone who doesn't know that.
**How to apply:** don't reintroduce batching by pushing grid decisions back into the models. See
[[hf-nifti-wrapper-reorients-wrong]], [[seg-probe-design]].
