---
name: hf-nifti-wrapper-reorients-wrong
description: Use nanobrain's nifti.canonical_img, never HF datasets' Nifti1ImageWrapper — the latter reorients incorrectly.
metadata:
  type: project
  observed: 2026-07-27
---

HF `datasets`' `Nifti1ImageWrapper` reorients incorrectly. Always use `nifti.canonical_img`.

**Why:** a wrong reorientation is silent — the volume still looks like a brain — and it corrupts the
grid contract every model and the seg probe depend on.
**How to apply:** it is the one-line rule for any new task or wrapper that touches a nifti. See
[[eval-interface-is-nifti-in]].
