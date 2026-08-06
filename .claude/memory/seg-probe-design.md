---
name: seg-probe-design
description: How probe_seg.py is built — patch point cloud in world mm, nearest-patch assignment, subject-level repeated CV, one forward pass holding every subject's features.
metadata:
  type: project
  observed: 2026-08-06
---

Rebuilt on the patch contract, replacing the per-voxel `dense_embed` design:

- the model returns `PatchFeatures(features (N, D), coords (N, 3))`, coords in **RAS world mm of the
  input image**; the model never sees the label grid;
- each in-brain label voxel takes its **nearest patch** (`cKDTree`), so labels are never resampled
  and every backbone is scored on the task's own native grid whatever its patch size;
- nearest assignment makes predictions constant within a patch, so scoring runs the head over the
  N patches and gathers out to voxels, not over millions of voxels;
- subject-level repeated CV, generic over K foreground classes (labels 1..K, 0 = background),
  scored by per-subject Dice (argmax) and voxel-AP; empty ground truth scores specificity in Dice
  and NaN in AP;
- **one forward pass**, with every subject's patch features held in memory at once.

**Why:** the old two embed passes existed purely to bound peak memory near one volume's embedding,
which was 24 GB for a 768-d ViT on the task-4 grid. Patch features are ~2 MB, so that bound stopped
mattering and the second pass went. This is a deliberate reversal of the previous note, not drift.

**How to apply:** memory is now linear in cohort size rather than bounded — fine for the FOMO seg
tasks (tens of subjects), but `random_unet` at `patch=4` emits 128-d at stride 4, roughly 64 MB per
subject on the task-4 grid. If a large-cohort seg task arrives, re-split the passes rather than
shrinking the cohort. Keeping exactly one extraction call site is also the seam where batched
inference would drop back in. See [[eval-interface-is-nifti-in]], [[seg-probe-world-coord-guards]].
