---
name: hf-revision-ignored-by-packaged-builders
description: load_dataset(revision=...) is silently ignored by the packaged arrow/parquet builders — a garbage revision still loads; pin with the @rev URL form so a bad pin fails loudly.
metadata:
  type: project
---

`load_dataset(revision=...)` only applies when loading by repo id. The packaged `arrow`/`parquet`
builders accept the kwarg and **ignore it silently** — verified: a garbage revision still loaded all
1,000 rows of `medarc/ppmi-mini`.

Put the revision in the path instead, where a bad one fails loudly:

```
hf://datasets/<repo>@<rev>/data/eval-*.parquet
```

On the plain repo-id path the kwarg *is* honored.

**Why:** a pin that silently does nothing is worse than no pin — it reads as reproducible in the
code and isn't. Compounds with [[adni-mini-reuploaded-in-place]], where upstream changed under a
fixed repo id.
**How to apply:** whenever the parquet-glob workaround is in use, the `@rev` form is mandatory. See
[[hf-readme-overrides-parquet-schema]].
