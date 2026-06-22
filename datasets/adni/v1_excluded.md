# ADNI evals — excluded from v1

This file records what is **deliberately not** in the v1 ADNI eval suite, and why,
so the scope is explicit and the deferred items are easy to pick up later.

## v1 scope (for reference)

Cross-sectional, single-scan-input tasks tied as closely as possible to the
underlying AD biomarkers:

1. AD vs CN diagnosis (ADNIMERGE `DX` / `DX_bl`) — binary
2. Amyloid-PET positivity (UC Berkeley `AMYLOID_STATUS`) — binary
3. Amyloid burden (UC Berkeley `CENTILOIDS` / `SUMMARY_SUVR`) — regression
4. Tau-PET positivity (`META_TEMPORAL_SUVR` threshold) — binary
5. Tau burden (`META_TEMPORAL_SUVR`) — regression
6. CSF Aβ42 (ADNIMERGE `ABETA`) — regression
7. CSF p-tau (ADNIMERGE `PTAU`) — regression
8. CSF t-tau (ADNIMERGE `TAU`) — regression
9. MCI → AD conversion (ADNIMERGE `DX` trajectory) — time-to-event

Every v1 task uses a **single scan as input**. Tasks 1–8 are concurrent
(diagnosis, or a scan paired to a contemporaneous biomarker); task 9 is
prognostic but still single-scan-input — the time-to-event structure lives only
in the label.

## Excluded from v1

### 1. Slope-based longitudinal tasks
- **Cognitive decline** — per-subject slope of CDRSB (or ADAS13 / mPACC) over
  repeated visits.
- **Longitudinal atrophy** — per-subject annualized % change in hippocampal
  (or ventricular / whole-brain) volume across imaging sessions.

**Why excluded:** these targets require collapsing each subject's *multiple*
follow-up measurements into a rate. The data exists (~2,000 subjects with ≥2
timepoints), and the model could in principle predict the slope from a single
baseline scan, but we are scoping v1 to the direct biomarker readouts and the
one conversion task. These are the natural first additions to a later version.

**Note:** MCI → AD conversion (v1 task 9) is *also* derived from longitudinal
follow-up, but it is kept in v1 because (a) it is a discrete clinical endpoint
rather than a fitted slope and (b) it is the ADNI prognostic task used in the
reference paper (Neuro-JEPA, arXiv:2606.14957).

### 2. Plasma biomarkers
- Plasma p-tau181 / p-tau217, NfL, GFAP, plasma Aβ42/40.

**Why excluded:** not present in any file currently downloaded (ADNIMERGE has no
plasma columns; the PET zips contain none). Requires a separate LONI download
(e.g. UGOT plasma p-tau, Fujirebio Lumipulse p-tau217, Blennow NfL/GFAP panels).
Planned for v2.

### 3. Longitudinal *input* modeling
- Feeding the model a pair/sequence of scans to measure change directly
  (e.g. baseline + follow-up → atrophy), Siamese/delta heads, temporal attention.

**Why excluded:** the current model treats every scan as independent — there is
no architecture for paired or temporal inputs. All v1 tasks are therefore
single-scan-input. Multi-scan input is a v2+ modeling change, distinct from the
slope-based *labels* in exclusion (1), which only need single-scan input.
