---
name: plain-matplotlib-for-internal-plots
description: Internal plots should be plain matplotlib with default colors and short readable code — the full design treatment is only for figures meant to be published.
metadata:
  type: feedback
---

The eval radar was built to a full design spec — validated palette, CVD secondary encoding, dataset
wedges, per-spoke chance arcs. Connor: *"the data viz skill is a bit intense for my liking… ideally
the first version should be short and easily readable code."*

Internal plots: default matplotlib colors, no annotations or custom palettes, short readable code.
Save the full treatment for figures that are actually going to be published.

**Why:** he reads the plotting code as often as the plot, and an internal figure is a measurement,
not a deliverable.
**How to apply:** first version is always the plain one; escalate only when he says the figure is
for publication. This overrides the `dataviz` skill's defaults for internal work. See
[[over-producing-the-artifact]].
