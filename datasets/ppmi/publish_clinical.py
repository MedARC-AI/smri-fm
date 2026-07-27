"""Open a pull request on medarc/ppmi-mini adding the clinical target columns.

The imaging in ppmi-mini never changes; this only adds a small parquet of
derived clinical columns keyed by sample_id, so the arrow dataset does not need
regenerating and the upload is ~120 KB rather than many GB.

Requires a WRITE-scoped HF token. Opening a PR does not require write access to
the repo itself, but it does require the token to carry write scope:

    hf auth login          # paste a write token, or
    HF_TOKEN=... uv run python datasets/ppmi/publish_clinical.py --dry-run

Run --dry-run first: it prints exactly what would be uploaded and exits.
"""

import argparse
from pathlib import Path

import pandas as pd

REPO_ID = "medarc/ppmi-mini"
PATH_IN_REPO = "clinical/ppmi_mini_clinical.parquet"

PR_TITLE = "Add clinical target columns (UPDRS, MoCA, cognitive composite, PRS)"

PR_BODY = """\
Adds a small parquet of derived clinical targets for the existing 1000 scans, joined from PPMI LONI study data on PATNO. `participant_id` is `sub-<PATNO>`, so the join is 1000/1000 and no imaging was reprocessed.

The file is keyed on `sample_id` and lives at `clinical/ppmi_mini_clinical.parquet`, so nothing about the arrow dataset changes. Load it alongside the split and align on `sample_id`.

Columns are baseline (value nearest the scan) and `slope_48m` (annualized OLS slope over visits in a −0.25 to 4.0 year window, minimum 2 visits), plus `n_visits` for each:

| Prefix | Instrument | Direction |
|---|---|---|
| `np3tot_off` / `np3tot_all` | MDS-UPDRS Part III motor exam. `_off` keeps OFF-medication and untreated exams; `_all` keeps every state | higher = worse |
| `nhy_off` / `nhy_all` | Hoehn & Yahr stage, 0–5. The out-of-range `101` "not assessed" code is mapped to null | higher = worse |
| `np2ptot` | MDS-UPDRS Part II, patient-reported daily living | higher = worse |
| `mseadlg` | Modified Schwab & England ADL, percent independence | higher = better |
| `mcatot` | MoCA, 0–30 | higher = better |
| `cogcomp` | Cognitive composite: mean z-score over HVLT-R, Symbol Digit, Benton JLO, Letter–Number Sequencing and semantic fluency, requiring ≥3 of 5 tests at a visit. Raw totals, not PPMI's age-normed T-scores | higher = better |

Single-timepoint columns: `upsit_baseline` (smell identification, n=386), `rbd_baseline` (REM sleep behaviour disorder screen, n=990), `prs_meta5` (META5 PD polygenic risk score, n=827).

`PATNO` is deliberately not included — `sample_id` already encodes it.

Note on `cogcomp`: MoCA ceilings badly in this cohort, with 66.7% scoring ≥27/30 and 12.2% at exactly 30, which is why the composite was added.

Caveat worth recording: using `prs_meta5` as a negative control, a frozen MAE encoder reaches r≈0.17 (ViT-B) and r≈0.21 (ViT-L) against a permutation null of −0.008±0.042. Not explained by age, sex, diagnosis, self-reported race, or fold leakage. Site or scanner is the leading suspect, and this dataset carries no scanner metadata to test it with.

This is subject-level PPMI data, so the repo needs to stay gated — the PPMI DUA forbids redistribution to anyone not independently registered with LONI.
"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--parquet",
        type=Path,
        default=Path("/mnt/data/medarc/datasets/ppmi/derived/ppmi_mini_clinical.parquet"),
    )
    p.add_argument("--dry-run", action="store_true", help="print and exit, upload nothing")
    args = p.parse_args()

    df = pd.read_parquet(args.parquet)
    assert "PATNO" not in df.columns, "PATNO must not be published; rebuild the parquet"
    assert len(df) == 1000, f"expected 1000 rows, got {len(df)}"

    size_kb = args.parquet.stat().st_size / 1024
    print(f"file      {args.parquet}  ({size_kb:.0f} KB)")
    print(f"target    {REPO_ID}:{PATH_IN_REPO}")
    print(f"rows      {len(df)}   columns {len(df.columns)}")
    print("coverage:")
    for c in sorted(df.columns):
        if c == "sample_id" or c.endswith("_n_visits"):
            continue
        print(f"  {c:26s} n={df[c].notna().sum():4d}/1000")

    if args.dry_run:
        print("\n--dry-run: nothing uploaded")
        return

    from huggingface_hub import HfApi

    commit = HfApi().upload_file(
        path_or_fileobj=str(args.parquet),
        path_in_repo=PATH_IN_REPO,
        repo_id=REPO_ID,
        repo_type="dataset",
        create_pr=True,
        commit_message=PR_TITLE,
        commit_description=PR_BODY,
    )
    print(f"\nPR opened: {commit.pr_url}")


if __name__ == "__main__":
    main()
