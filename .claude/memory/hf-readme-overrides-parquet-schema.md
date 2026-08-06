---
name: hf-readme-overrides-parquet-schema
description: load_dataset builds its schema from the README YAML dataset_info, not the parquet shards — an under-declared README makes every shard fail to cast; bypass via the parquet builder.
metadata:
  type: project
  observed: 2026-07-30
---

`load_dataset("<repo>")` builds its target schema from the README YAML `dataset_info.features`, not
from the metadata embedded in the parquet shards. If the README under-declares columns, every shard
fails to cast with `CastError: ... because column names don't match`, naming only the declared
subset as the target — so the error points at the shards when the README is at fault.

Bypass by loading the shards as a plain parquet dataset, which reads the embedded schema and
recovers `ClassLabel` and `Nifti` types:

```python
load_dataset("parquet", data_files={"eval": "hf://datasets/<repo>/data/eval-*.parquet"}, split="eval")
```

**Why:** the failure names the wrong culprit, so it costs time on the shard side before anyone reads
the README.
**How to apply:** prefer fixing the upstream README and returning to a plain `load_dataset`; the
bypass has its own trap, see [[hf-revision-ignored-by-packaged-builders]].
