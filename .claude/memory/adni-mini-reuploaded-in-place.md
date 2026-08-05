---
name: adni-mini-reuploaded-in-place
description: medarc/adni-mini was replaced in place on 2026-07-29 — brain-masked images, 1000 -> 1200 scans, split renamed; any ADNI number from before that date is not comparable.
metadata:
  type: project
---

Replaced on 2026-07-29 (sha `629f32de`) by a local derivative of `medarc/adni-mini-v1-3`. Three
things changed at once:

- split renamed `test` -> `eval`, and 1000 -> 1200 scans;
- images brain-masked (`image[synthseg_dseg == 0] = 0.0`), so voxels outside the SynthSeg brain are
  exactly zero — **a real change to model input**, not just packaging;
- the README's `dataset_info` was broken, which is what actually broke loading (see
  [[hf-readme-overrides-parquet-schema]]).

Task sizes after: age/sex 1200, ad_cn 750, amyloid_centiloid 997, tau_suvr 555.

Fixed upstream in `3fdea48` (2026-07-29), which touched only the README — shards byte-identical to
`629f32d` — so `tasks/adni.py` is back on a plain `load_dataset(revision=...)`.

**Why:** an in-place re-upload under a stable repo id means old and new numbers look comparable and
are not.
**How to apply:** treat any ADNI result computed before 2026-07-29 as unmasked images on a different
cohort size; do not put it in the same table as current numbers.
