---
name: verifying-patch-world-coords
description: How to check a backbone's patch_embed coords are right — the marker test must target a local layer, and which statistic to use per architecture. Learned the hard way on all four ports.
metadata:
  type: project
  observed: 2026-08-06
---

Porting a backbone to `patch_embed` means claiming its patch coords describe the image region its
features came from. That claim is **invisible to every geometric check** — see
[[seg-probe-world-coord-guards]] — so it has to be tested against content: perturb the image at a
known place, and see which patch's feature moves.

**The trap, hit on the first port (Neuro-JEPA).** Self-attention is global, so a local change
perturbs every output token, and the most-changed *output* token is not the one holding the marker.
Measured: 133mm mean error, and two different markers selected the *same* token — which is what
gave it away. Run the marker test against the **patch-projection layer, before attention**: the
same test then gave 17mm against a 21mm token pitch, i.e. correct.

**Pick the statistic to match the architecture:**

| backbone | layer to probe | statistic | why |
|---|---|---|---|
| ViT (Neuro-JEPA, NeuroVFM, sMRI MAE) | patch projection / `tokenize_volume`, pre-attention | argmax of feature delta | correspondence is exactly local there |
| conv U-Net (SynthSeg) | bottleneck | change-weighted **centroid** | receptive field is ~125 input voxels, so argmax wanders up to 27mm — more than a patch |

**Two phantom gotchas.** A *bright* marker is useless wherever preprocessing does percentile
intensity scaling (SynthSeg): it becomes a global shift and every marker position returns the same
answer — use a dark one. And NeuroVFM drops any token holding a sub-threshold voxel, so a noisy or
centred phantom yields zero tokens.

**Make the phantom asymmetric and off-centre.** A centred cube is symmetric under transposition and
scores 1.000 with transposed coords, catching nothing.

**How to apply:** for a purely geometric preprocessing chain (sMRI MAE's rescale + centre pad/crop,
SynthSeg's resample + align + pad), the stronger check is Connor's: push a volume whose voxel
*values* are its own world coordinates through the chain and confirm each output voxel holds the
coordinate the affine predicts. That validated to 2.9e-6 mm and needs no reasoning about
`align_corners` or pad-offset signs. Use the marker test for what that cannot reach — token
ordering and the flatten/coord pairing. Belongs in `.claude/skills/add-eval-model` if a fifth
backbone is ever ported.
