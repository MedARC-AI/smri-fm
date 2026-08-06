---
name: flash-attn-rejected
description: flash-attn was evaluated and rejected for NeuroVFM — the prebuilt wheel installs fine but omits the fused_dense_lib CUDA extension, which needs nvcc and a second build pass.
metadata:
  type: project
  observed: 2026-07-29
---

Prebuilt wheels exist on GitHub releases for exactly cp311/torch2.8/cu12/cxx11abiTRUE and install in
21s (PyPI has only an sdist; `torch._C._GLIBCXX_USE_CXX11_ABI` picks the variant).

The blocker is not the wheel. `FusedDense`/`FusedMLP` live in the `fused_dense_lib` CUDA extension,
which **is not in that wheel** — it builds from `csrc/fused_dense_lib` with `--no-build-isolation`
in ~4 min, so `uv sync` cannot do it in one pass and nvcc is needed on whatever machine syncs. That
fragility is the reason for the rejection.

`from flash_attn.modules.mlp import FusedMLP` *is* a valid import path — it re-exports from
`ops.fused_dense` — it just resolves to `None` without the extension, so an import guard is not
enough to detect it.

**Why:** the failure mode looks like a working install right up until the ops resolve to `None`.
**How to apply:** stay on the pure-torch path; it is not an approximation, see
[[neurovfm-torch-fallback-verified]].
