# ADNI evaluation dataset

`curate.py` reproduces the ADNI evaluation cohort published at
`medarc/adni_eval`. It matches processed scans to ADNIMERGE records, applies
quality and demographic filters, constructs subject-exclusive balanced splits,
and writes Arrow shards with embedded NIfTI bytes.

This script is not used during evaluation. Evaluation loads the published dataset
directly with `load_dataset("medarc/adni_eval", token=True)`.

```bash
uv run python datasets/adni/curate.py \
  --data-root /path/to/adni \
  --clinical-csv /path/to/ADNIMERGE.csv \
  --output-dir /path/to/adni-smri \
  --num-proc 8 --max-shard-size 1GB
```
