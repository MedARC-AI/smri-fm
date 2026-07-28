# Eval

Frozen-feature sklearn probes for structural-MRI backbones. A backbone is evaluated by
extracting features once, then fitting a small sklearn head under repeated cross-validation.

## Run

```bash
uv run python -m nanobrain.eval.main <model> <task> [--config cfg.yaml] [--overrides key=value ...]
# e.g. the random-features baseline on brain age
uv run python -m nanobrain.eval.main random_features fomo_task3_age
```

`model` and `task` are registered names. Run settings come from
[config.yaml](config.yaml), overridden by `--config` then dot-list `--overrides`.

Outputs land in `<output_root>/<model>__<task>/`: `metrics.jsonl`, `config.yaml`, `log.txt`.

## Probes

Chosen by task type; each fixes its own metrics.

| Task type | Feature | Head | Metrics |
|---|---|---|---|
| `RegressionTask` | global embedding | `RidgeCV` | MAE, Pearson r |
| `ClassificationTask` | global embedding | `LogisticRegressionCV` | AUROC, balanced accuracy |
| `SegmentationTask` | dense embeddings | `LogisticRegression` (balanced) | Dice, voxel-AP |

Segmentation is voxel-level: the head is fit on subsampled voxels (every foreground voxel plus a
capped draw of in-brain background) and scored over every in-brain voxel of held-out subjects,
per foreground class. Cross-validation splits subjects, not voxels.

Cross-validation is `n_splits`-fold repeated `n_repeats` times. Metrics pool out-of-fold
predictions over all samples within a repeat, then average across repeats — so rank metrics
see all N test points. The reported std is the spread across repeats: a rough stability
signal, not a confidence interval (it understates true sampling variance).

## Adding things

- **Model**: an `nn.Module` implementing the nifti-in contract in
  [models/base.py](models/base.py) (`global_embed`, `dense_embed`), canonicalizing and
  normalizing each volume itself with the helpers in [nifti.py](nifti.py). Decorate a builder
  with `@register_model`. See [models/random_features.py](models/random_features.py) and
  [models/unet.py](models/unet.py).
- **Task**: a dataclass from [tasks/base.py](tasks/base.py) wrapping a lazy `dataset_fn`
  (an HF dataset of niftis + labels) and column names. Decorate a builder with
  `@register_task`. See [tasks/fomo.py](tasks/fomo.py).
