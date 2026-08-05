---
name: sdpa-not-faster-at-real-token-counts
description: Swapping NeuroVFM's materialized attention matrix for SDPA is 1.6x at N=500 but 0.9x at the real ~2000 tokens — kept for memory, not speed.
metadata:
  type: project
---

Swapped the `use_flash_attn=False` path off its materialized `(B,H,N,N)` score matrix (cosine
0.999995). Honest result: at real token counts (~2000) it is **not a speedup** — 1.6x at N=500,
0.9x at N=2000 — because the qkv/proj linears dominate, not the attention.

Kept anyway, for memory and scaling headroom.

**Why:** the microbenchmark at a small N pointed the opposite way from the real workload.
**How to apply:** benchmark attention swaps at the token count the model actually runs at, and
report the negative result rather than the favourable N.
