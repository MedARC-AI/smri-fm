---
name: hf-from-generator-cache-ignores-code
description: HF Dataset.from_generator keys its cache on gen_kwargs only — editing a generator body or a module-level global silently reuses stale cached data; Features changes are the one exception.
metadata:
  type: project
---

`Dataset.from_generator` (datasets 5.0.0) keys its cache dir on `gen_kwargs` only. Module-level
globals the generator reads are **not** hashed — dill pickles importable functions by reference, so
changing `fomo.BASE_URL` leaves `Hasher.hash(fomo._generate_task1)` identical, and the
`FOMO_EVAL_BASE_URL` / `FOMO_EVAL_TASK5_URL` overrides silently reuse whichever source was cached
first.

Confirmed the hard way (2026-07-27): after fixing seg/image grid alignment, both
`fomo_task1_infarct_seg` and `fomo_task2_meningioma` reloaded pre-fix data and Task 2 failed again
with the same assertion. Only `rm -rf $HF_HOME/datasets/generator/default-<hash>` forced a rebuild.

**One exception**, verified 2026-07-28 while adding `cnp.py`: the `Features` schema *is* part of the
fingerprint, so adding a `scanner` column triggered a full rebuild with no manual purge. Schema
changes are safe; changes to generator bodies and preprocessing helpers are not.

A durable fix would thread a `version` value through `gen_kwargs` and bump it when preprocessing
changes.

**Why:** it fails silently and looks like the fix didn't work, which sends you back into the code
that was already correct.
**How to apply:** purge the generator cache after any edit under `tasks/` before believing a rerun.
See [[hf-readme-overrides-parquet-schema]].
