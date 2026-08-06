---
name: nested-tensor-sdpa-needs-a-device
description: Nested-tensor SDPA raises AcceleratorError on a CUDA-built torch with no visible device, so sMRI MAE cannot forward on the login node — tests build a depth-0 encoder instead.
metadata:
  type: project
  observed: 2026-07-29
---

The sMRI MAE blocks use nested-tensor SDPA, whose backend selection calls
`torch._C._can_use_cudnn_attention`. On a CUDA-built torch with no visible device that **raises**
rather than falling back, so the forward pass will not run on the login node at all.

Workaround used in tests: build a **depth-0** encoder. That keeps patchify, masking, pos embed and
pooling — everything the wrapper is actually responsible for — while skipping every attention block.

Real forward passes: 0.13 s/volume warm, 2.1 GB peak at 208x240x208.

**Why:** it is a hard import-time-ish failure, not a slow path, so "just run it on CPU" is not
available.
**How to apply:** reach for the depth-0 trick for any transformer whose CPU path is blocked; it
covers the wrapper's own logic without a GPU. Use the `gpu-session` skill for real allocations.
