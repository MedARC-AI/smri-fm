---
name: smri-mae-pad-to-multiple-is-inert
description: The sMRI MAE encoder's pad_to_multiple cannot change any token value — padded slots are dropped before the blocks — so the wrapper stops passing it.
metadata:
  type: project
---

`SmriMae` used to forward `pad_to_multiple` (32, read from the checkpoint args) into the encoder.
Removed 2026-08-06. It only rounds the token-slot count up (`masking.py:patch_ids_from_mask`), and
`model_mae.py:255-256` then does

```python
jagged_batch = JaggedBatch.from_mask(token_mask)
x = x[token_mask]
```

so the padded slots are dropped *before* the transformer blocks and attention runs jagged over the
real tokens only. It is a sequence-alignment knob for batched training, inert at batch size 1.

**Why:** it reads like a pretraining-fidelity setting that must be matched, and it is not — it
cannot move a number. It also confused what `patch_embed` had to do: the padding is on the **token
sequence**, not the volume, so the token grid is exactly `img_size // patch_size` and the grid
still tiles the fitted volume.

**How to apply:** don't reintroduce it for faithfulness reasons. If batching ever returns, it
becomes relevant again as an efficiency knob only. Verified by reading rather than measuring —
depth > 0 cannot run on a CPU-only node, since the jagged SDPA asks the cuDNN backend whether it
can run. See [[smri-mae-preprocessing-gap]].
