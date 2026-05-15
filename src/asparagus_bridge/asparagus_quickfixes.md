# Asparagus quickfixes

This file documents the local changes currently present in the `third_party`
submodules compared with their checked-out upstream versions. These approapriate fixes are submitted as PRs to the upstream versions, but until they are merged, you can just tell an LLM to make these changes in your local code.

## `third_party/asparagus`

### `configs/core/base.yaml`

- Line 13: adds `checkpoint_path` to Hydra's `override_dirname.exclude_keys`.
  This keeps long checkpoint paths out of Hydra-generated override directory
  names.
- Lines 35-36: comments out the original `hydra.run.dir` template and replaces
  it with a shorter version that removes the `__clargs=${hydra.job.override_dirname}`
  path component.
- Lines 39-40: applies the same shortening to `hydra.sweep.subdir`, preserving
  the original template as a comment and using a replacement without the
  `__clargs=${hydra.job.override_dirname}` component.

Overall effect: Hydra output paths are shorter, especially when checkpoint
paths or many command-line overrides would otherwise make run directories too
long.
Without the changes, if running fine-tuning with CLI arguments instead of config, you will likely get the error `filename too long` (on MacOS and Linux).

## `third_party/asparagus_preprocessing`

### `asparagus_preprocessing/utils/splitting.py`

- Lines 10-11: adds `split_80_10_10(files, test=False, seed_increment=0)`.
  The new helper calls `non_stratified_split` with train/validation/test
  fractions of `0.80`, `0.10`, and `0.10`, using base seed `28300211`.

### `asparagus_preprocessing/datasets_classification/CLS002_FOMO26_Infarct.py`

- Line 88: keeps `split="split_80_10_10"` as the active dataset split.
- The duplicate later `split=None` argument was removed, so the configured
  split is no longer overwritten.

### `asparagus_preprocessing/datasets_classification/CLS003_FOMO26_Polymicrogyria.py`

- Line 76: keeps `split="split_80_10_10"` as the active dataset split.
- The duplicate later `split=None` argument was removed, so the configured
  split is no longer overwritten.

### `asparagus_preprocessing/datasets_regression/REGR002_FOMO26_BrainAge.py`

- Line 76: keeps `split="split_80_10_10"` as the active dataset split.
- The duplicate later `split=None` argument was removed, so the configured
  split is no longer overwritten.

### `asparagus_preprocessing/datasets_segmentation/SEG009_FOMO26_Meningioma.py`

- Line 80: keeps `split="split_80_10_10"` as the active dataset split.
- The duplicate later `split=None` argument was removed, so the configured
  split is no longer overwritten.

### `asparagus_preprocessing/datasets_segmentation/SEG010_FOMO26_TrigeminalNeuralgia.py`

- Line 72: keeps `split="split_80_10_10"` as the active dataset split.
- The duplicate later `split=None` argument was removed, so the configured
  split is no longer overwritten.

Overall effect: the FOMO26 classification, regression, and segmentation
preprocessing scripts now request the newly added `split_80_10_10` function
instead of accidentally disabling splitting with a duplicate `split=None`
argument.
