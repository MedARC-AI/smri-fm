---
name: seg-probe-design
description: How probe_seg.py is built — subject-level repeated CV, generic over K foreground classes, two embed passes to bound peak memory near one volume's embedding.
metadata:
  type: project
---

Rewritten alongside the nifti-in interface (`50cb2a8`):

- subject-level repeated CV, generic over K foreground classes (labels 1..K, 0 = background);
- scored by per-subject Dice (argmax, implicit 0.5) and voxel-AP;
- **two embed passes** — subsample for training, full-brain for scoring — to bound peak memory near
  one volume's embedding;
- subjects with empty ground truth score specificity in Dice and NaN in AP.

**Why:** the two-pass structure exists purely for the memory bound, so collapsing it to one pass
looks like a simplification and isn't.
**How to apply:** keep the two passes; it is untuned in other respects — `NEG_PER_SUBJECT=10_000`
and real-data peak memory were deferred on "measure first". See [[eval-interface-is-nifti-in]].
