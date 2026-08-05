---
name: ppmi-mini-confound-floors
description: PPMI-mini has no site or scanner column and its cohorts were enrolled in waves — measured confound floors, what shipped, and why Prodromal-vs-CN and SWEDD-vs-CN were rejected.
metadata:
  type: project
---

`medarc/ppmi-mini` v0.1 has no site or scanner column, so scan date is the only acquisition handle,
and the cohorts were enrolled in waves.

Unmatched confound AUROC floors:

| Split | Scan year | Header geometry |
|---|---|---|
| PD vs CN | 0.61 | — |
| Prodromal vs CN | 0.84 | 0.79 |
| PD vs Prodromal | 0.78 | 0.69 |
| SWEDD vs CN | — | 0.66 |

**Shipped:** `ppmi_age` (999), `ppmi_pd_cn` (426), `ppmi_pd_prodromal` (324), matched on (scan-year
band, age band, sex).

**Rejected:** Prodromal vs CN — matched pool only 146 and geometry still 0.57, because CN is
2010-2013 with a small 2021-2024 tail while prodromal is almost all 2018+, so they barely overlap in
time. SWEDD vs CN — 61 subjects all scanned 2011-2013, matching still leaves scan year at 0.73, and
the group is biologically mixed anyway. No `ppmi_sex` — the suite already has three.

One non-3D volume excluded: `sub-3200_ses-20101202_T1w`, (512, 512, 78, 2).

**Why:** these rejections will look like missing tasks to anyone reading the suite later.
**How to apply:** don't re-add the rejected splits without new acquisition metadata. Procedure for
new cohorts is in the `add-eval-dataset` skill.
