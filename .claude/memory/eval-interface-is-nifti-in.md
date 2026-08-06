---
name: eval-interface-is-nifti-in
description: The eval suite uses a uniform per-nifti model interface so segmentation scoring is independent of any model's patch grid — the accepted cost is that batched inference is gone.
metadata:
  type: project
  observed: 2026-08-06
---

`src/nanobrain/eval` moved from a `(model, transform)` contract to a uniform per-nifti interface:
`global_embed(nifti) -> (D,)` and `dense_embed(nifti) -> (X, Y, Z, D)` on the RAS-canonical grid.
Models canonicalize, normalize and own device placement internally. Shared `canonical()` in
`nifti.py` is the grid contract the seg probe aligns labels to. Commit `50cb2a8`.

Motivation: make segmentation scoring independent of any model's patch grid, so backbones are
comparable. **Accepted cost: batched inference is gone** — one volume at a time.

**Update 2026-08-06: that trade is dissolved, not merely re-taken.** `dense_embed` became
`patch_embed(nifti) -> PatchFeatures(features (N, D), coords (N, 3))` in world mm. What broke
comparability in the pre-`50cb2a8` design was `patchify_labels`, which let the *model* define the
label grid; patch-level *features* were never the problem. The probe now maps voxels to patches by
world coordinate, so every backbone is scored on the task's own grid while the model returns
patch-level output again. Batching can therefore come back without giving up comparability.

**Why:** the old note read as "batching is the price of comparability", which is no longer the
constraint and would otherwise block an easy win.
**How to apply:** if batching returns, keep coords in world mm (so a subject's result does not
depend on its batch), keep the probe consuming a *list* of per-subject results (N is ragged —
NeuroVFM drops background tokens), and add it behind the single extraction call site in
`probe_seg.seg_probe`. Still don't push grid decisions back into the models. See
[[hf-nifti-wrapper-reorients-wrong]], [[seg-probe-design]].
