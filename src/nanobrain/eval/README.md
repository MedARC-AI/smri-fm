# Eval

Frozen-feature sklearn probes for structural-MRI backbones: extract features once, then fit a
small sklearn head under repeated cross-validation.

## Run

```bash
uv run python -m nanobrain.eval.main <model> <task> [--config cfg.yaml] [--overrides key=value ...]
uv run python -m nanobrain.eval.main random_features fomo_task3_age  # e.g. brain age
```

`model` and `task` are registered names; run settings come from [config.yaml](config.yaml),
overridden by `--config` then dot-list `--overrides`. Outputs land in
`<output_root>/<model>__<task>/`: `metrics.jsonl`, `config.yaml`, `log.txt`.

## Probes

Chosen by task type; each fixes its own metrics.

| Task type | Feature | Head | Metrics |
|---|---|---|---|
| `RegressionTask` | global embedding | `RidgeCV` | MAE, Pearson r |
| `ClassificationTask` | global embedding | `LogisticRegressionCV` | AUROC, balanced accuracy |
| `SegmentationTask` | dense embeddings | `LogisticRegression` (balanced) | Dice, voxel-AP |

Segmentation fits on subsampled voxels (every foreground voxel plus a capped draw of in-brain
background) and scores every in-brain voxel of held-out subjects, per foreground class.

Cross-validation is `n_splits`-fold repeated `n_repeats` times, always splitting subjects. Metrics
pool out-of-fold predictions within a repeat, then average across repeats — so rank metrics see all
N test points. The reported std is the spread across repeats: a stability signal, not a confidence
interval.

## Adding things

- **Model**: an `nn.Module` implementing [models/base.py](models/base.py) (`global_embed`,
  `dense_embed`), preprocessing each volume with the helpers in [nifti.py](nifti.py). Decorate a
  builder with `@register_model`. See [models/random_features.py](models/random_features.py) and
  [models/unet.py](models/unet.py).
- **Task**: a dataclass from [tasks/base.py](tasks/base.py) wrapping a lazy `dataset_fn` (an HF
  dataset of niftis + labels) and column names. Decorate a builder with `@register_task`. See
  [tasks/fomo.py](tasks/fomo.py).
