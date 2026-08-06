---
name: smri-mae-axis-order
description: The sMRI MAE eval transform feeds native RAS (X, Y, Z); the (Z, Y, X) transpose was measured wrong and the flag is gone.
metadata:
  type: project
  observed: 2026-08-06
---

`models/smri_mae.py` used to carry a `transpose` flag defaulting to `False`,
with a TODO calling the shape convention a footgun.

Evidence that identity (native RAS `(X, Y, Z)`) is correct:

- Padding the MNI template to 208x240x208 and dicing against a stored pretraining brain mask gives
  **0.89 / 0.85 for identity vs 0.75 for the `(Z, Y, X)` swap**.
- The global sweep agreed independently: `vitl_fomo300_tr`, the transposed config, was the worst
  row in `experiments/eval_global_0728/figures/table.md`.

Connor confirmed, so the flag, the TODO and the `_tr` sweep entries were deleted rather than left
as a knob. Output dirs named `smri_mae_vitl_fomo300_tr__*` predate this and are still scored by
`analysis_table.py`; they are a historical record and cannot be regenerated from the current code.

**Why:** a defaulted-off flag for a setting we had already measured as wrong is a live footgun —
one override away from silently producing the worst config in the table.

See [[smri-mae-preprocessing-gap]] and [[smri-mae-checkpoint]].
