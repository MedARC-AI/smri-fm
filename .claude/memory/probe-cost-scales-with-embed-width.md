---
name: probe-cost-scales-with-embed-width
description: Probe wall-clock scales hard with embedding width — adni_age RidgeCV over 1000 subjects is 31s at 1024-d vs 605s at 3840-d.
metadata:
  type: project
---

`adni_age` RidgeCV over 1000 subjects: **31s at 1024-d** (`random_features`) vs **605s at 3840-d**
(`random_unet`). Roughly 20x for under 4x the width.

**Why:** feature width is a real wall-clock knob for the sklearn heads, not just a memory one, and
it is easy to treat a wider backbone as free.
**How to apply:** weigh output dim when picking or configuring a backbone, and expect wide-feature
sweeps to be dominated by probe time rather than by the forward pass.
