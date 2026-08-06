---
name: dice-ceiling-diagnostic-deferred
description: A per-(model, task) Dice ceiling would separate "too coarse" from "uninformative" in the seg probe — deferred as not strictly necessary; includes the exact formula and the ratio trap.
metadata:
  type: project
  observed: 2026-08-06
---

Under the patch-point-cloud contract ([[seg-probe-design]]) the seg probe assigns each
voxel its nearest patch, so predictions are piecewise-constant over patch Voronoi cells. A model
with 21 mm patches (Neuro-JEPA) cannot trace a trigeminal nerve **regardless of feature quality**,
so a low Dice conflates two different failures: coarse tessellation vs. uninformative features.

A **Dice ceiling** — the best Dice achievable on that model's tessellation, depending only on patch
coords and task labels, *not* on the features — would separate them. Computed once per
(model, task). Deferred: not strictly necessary, and raw Dice is comparable without it.

**Compute it exactly, not by majority vote.** Majority-vote-per-patch is worst precisely where the
diagnostic is most needed: with a 0.5 mm nerve under 21 mm patches no patch is majority-foreground,
so it returns 0 while the optimal labeling scores well above 0. With `t_i` foreground voxels in
patch `i` of `n_i` total and `T` foreground overall, selecting patches `S` gives
`Dice = 2·Σ_S t_i / (Σ_S n_i + T)`. Any maximizer is a superlevel set of `t_i / n_i` (Dinkelbach:
for fixed λ the optimum keeps exactly those with `t_i - λ·n_i > 0`), so sort by that fraction
descending and prefix-scan:

```python
order = np.argsort(-t / n)
ceiling = (2 * t[order].cumsum() / (n[order].cumsum() + T)).max()   # exact optimum
```

**Why:** the obvious implementation (majority vote) is wrong in the one regime that motivates the
metric, and the obvious way to *use* the result is also wrong — `dice / ceiling` is **not**
comparable across models, since a model with a tiny ceiling can score 1.0 on it while being useless.
That ratio trap is the main argument against shipping the ceiling at all.

**How to apply:** if this is revived, raw `dice` stays the comparable headline and `dice_ceiling` is
diagnostic only. Related: without it, a near-zero Dice from a genuinely coarse model looks identical
to one from a broken world-coord affine. The geometric guard only partly covers that gap — it is
blind to axis permutations by construction, so the asymmetric test phantom is the real backstop.
See [[seg-probe-world-coord-guards]].
