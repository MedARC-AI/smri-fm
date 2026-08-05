---
name: neurovfm-dependency-traps
description: NeuroVFM's eager __init__ drags the training and VLM stacks into a frozen-encoder install — outlines must pin to 1.1.1, torch_scatter comes from data.pyg.org.
metadata:
  type: reference
---

A lazy-`__init__` refactor was rejected as out of scope, so the eager import chain must be satisfied.
`neurovfm/__init__.py` pulls the training and VLM stacks, making pytorch-lightning, outlines, peft,
transformers, torchmetrics and openai import-time requirements for a frozen encoder.

- `outlines` must be pinned to **1.1.1** — 1.3.x moved `outlines.processors.structured`.
- `torch_scatter` has no PyPI wheel; data.pyg.org hosts `torch_scatter-2.1.2+pt28cu128-cp311`.
- Installing this set upgrades huggingface-hub 0.36 -> 1.25.

See [[neurovfm-integration-notes]].
