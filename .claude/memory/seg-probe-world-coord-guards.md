---
name: seg-probe-world-coord-guards
description: What the seg probe's coverage assert can and cannot catch — measured: units and origin yes, axis permutations never, and why the guard scales by cell volume not nearest-neighbour spacing.
metadata:
  type: project
  observed: 2026-08-06
---

`assign_patches` asserts `median(voxel -> nearest patch) < 2 * cell`, where
`cell = (prod(ptp(coords)) / N) ** (1/3)` — the edge of the cell the patch cloud implies.

**Measured** on a 160³ head phantom, ratio of median voxel distance to the guard's scale (fires at
2.0). Two candidate scales: `nn` = median nearest-neighbour distance between patches (the first
implementation), `cell` = the volume-derived one that shipped.

| coord mutation | ratio (nn) | ratio (cell) |
|---|---|---|
| correct | 0.51 | 0.43 |
| axis swap (x↔y, x↔z) | 0.51 **blind** | 0.43 **blind** |
| corner instead of centre | 0.54 blind | 0.45 blind |
| one-patch translation | 0.51 blind | 0.43 blind |
| sign flip | 5.6 fires | 4.7 fires |
| affine translation dropped | 4.7 fires | 3.9 fires |
| cm instead of mm | 79.7 fires | 67.0 fires |
| **anisotropic grid (1,1,8) mm** | **2.03 FALSE FIRE** | 0.88 passes |
| anisotropic grid (1,1,12) mm | 3.04 FALSE FIRE | 1.12 passes |

**Why cell volume, not nearest-neighbour spacing:** on an anisotropic grid the NN distance is the
*smallest* axis pitch while the voxel-to-patch distance follows the largest, so the ratio grows with
anisotropy and aborts legitimate runs. `tasks/fomo.py` resamples with `np.maximum(native,
min_spacing)`, which *preserves* native anisotropy, so a 0.9×0.9×5 mm DWI or FLAIR would have
tripped it. The volume-derived cell edge is the geometric mean of the pitches and is invariant.

**Why an axis permutation is undetectable here:** on a cubic grid the permuted point cloud is the
identical *set* of points, so no purely geometric statistic can see it. Only content can — hence
`test_probe_seg.BOXES` places its phantom boxes so that swapping any two axes lands each box
entirely on background (measured: Dice 1.000 correct vs 0.078 transposed). A symmetric cube fixture
scored 1.000 under transposition and caught nothing. Port a new backbone against an off-centre,
axis-asymmetric phantom, not a centred cube. See [[seg-probe-design]],
[[dice-ceiling-diagnostic-deferred]], [[smri-mae-axis-order]].
