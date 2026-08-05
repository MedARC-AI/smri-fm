---
name: synthseg-no-crop-by-default
description: SynthSeg's default inference path does not crop — the help string's "Default is 192" is stale and wires nothing; a brain-centred crop was built then removed to match upstream.
metadata:
  type: project
---

`--crop` has no argparse default, so it is `None` and `min_pad=128`. The help string's "Default is
192" is stale and wires nothing.

A brain-centred 192 crop was built and then **removed** to match their real default — Connor's call.
Worth knowing it changed features a lot: cosine 0.83-0.91 vs uncropped.

**Why:** matching upstream behaviour beats matching upstream documentation, and the two disagreed.
**How to apply:** if the crop question comes back, this is a deliberate decision, not an oversight —
reopen it explicitly rather than treating the missing crop as a bug. See
[[synthseg-pooling-masks-padding]], which is the mitigation that was kept.
