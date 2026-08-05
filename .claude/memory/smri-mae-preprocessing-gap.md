---
name: smri-mae-preprocessing-gap
description: sMRI MAE's eval transform uses a mean-intensity threshold as a stand-in for SynthSeg brain masking, so it keeps skull and neck that pretraining never saw — the main known fidelity gap.
metadata:
  type: project
---

Pretraining preprocessing (`src/preprocessing/pipeline.py` in `/data/connor/smri-fm`, plus
`datasets/FOMO300/metadata.json`): ANTs *rigid* registration to TemplateFlow `MNI152NLin2009cAsym`
at 1mm (193x229x193), SynthSeg brain mask, per-sample z-score over masked brain voxels *after* the
shape fit to 208x240x208, stored as fp16 brain-only voxels plus a bit-packed mask.

The eval transform's mean-intensity threshold is a stand-in for SynthSeg and keeps skull and neck,
which pretraining never saw. That is the main known fidelity gap.

To re-check the axis order, read a shard directly out of
`/data/connor/smri-fm/data/FOMO300/train/*.tar` — keys `image_values.npy`, `img_mask.npy`,
`meta.json`.

**Why:** with no upstream inference path there is nothing to equivalence-test against, so the
stand-ins have to be written down or they become invisible assumptions.
**How to apply:** when comparing sMRI MAE against other backbones, attribute part of any deficit
here before concluding anything about the model. See [[smri-mae-checkpoint]].
