# ADNI evaluation dataset

`curate.py` reproduces the ADNI v1 evaluation cohort published at
`medarc/adni_eval`. It matches processed scans to the raw ADNI study-data tables
(PTDEMOG, DXSUM, UPENN Roche Elecsys CSF) and the UC Berkeley amyloid/tau PET
tables, applies quality and demographic filters, attaches the nine v1 task labels
(see `v1_tasks.md`), assigns one subject-exclusive split, and writes Arrow shards
with embedded NIfTI bytes. ADNIMERGE (the stale derived table) is not used.

This script is not used during evaluation. Evaluation loads the published dataset
directly with `load_dataset("medarc/adni_eval", token=True)` and pools the splits,
running grouped CV on `participant_id`.

```bash
uv run python datasets/adni/curate.py \
  --data-root /path/to/adni \
  --demog-csv /path/to/PTDEMOG.csv \
  --dxsum-csv /path/to/DXSUM.csv \
  --csf-csv /path/to/UPENNBIOMK_ROCHE_ELECSYS.csv \
  --amyloid-csv /path/to/UCBERKELEY_AMY_6MM.csv \
  --tau-csv /path/to/UCBERKELEY_TAU_6MM.csv \
  --output-dir /path/to/adni-smri \
  --num-proc 8 --max-shard-size 1GB
```

- `v1_tasks.md` — the nine tasks, their label sources, and per-task cohorts.
- `v1_excluded.md` — what is deliberately not in v1 (plasma, slope-based
  longitudinal tasks, multi-scan input modeling) and why.
