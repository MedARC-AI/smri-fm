# `eval` — sMRI representation evaluation

Probes a frozen pretrained sMRI backbone on downstream tasks. Two evaluation modes
share the same model/dataset abstractions:

- **`main_linear`** — extract features once, fit a closed-form linear head
  (logistic / ridge) and sweep its regularization on the curated validation split.
  
- **`main_probe`** — train a small classifier head (linear / MLP / attention-pool)
  with SGD over a learning-rate × weight-decay grid, early-stopping on the validation
  split. 

Both keep the backbone **frozen** and report metrics on `train` / `validation` / `test`.

## Layout

```
eval/
├── main_linear.py        # linear-probe entrypoint (cli)
├── main_probe.py         # trained-head entrypoint (cli)
├── classifiers.py        # head architectures + ClassifierGrid (probe only)
├── utils.py              # metrics, preprocessing, config, shared helpers
├── config/
│   ├── default_linear.yaml
│   └── default_probe.yaml
├── models/               # backbone plugins (registry + smri_mae)
└── datasets/             # dataset plugins (registry + adni)
```

`models/` and `datasets/` auto-discover plugins at import: any module dropped in
either package that calls `@register_model` / `@register_dataset` becomes available
by name. `list_models()` / `list_datasets()` drive the CLI `choices`.

## Available pieces

| | names |
|---|---|
| **models** | `smri_mae` |
| **datasets** | `adni_sex`, `adni_ad_cn` (classification), `adni_age`, `adni_synthseg_volumes` (regression) |
| **representations** | `cls`, `patch` (patch tokens are mean-pooled to one vector) |
| **classifiers** (probe only) | `linear`, `mlp`, `attn` |

## Running

Both entrypoints are exposed as console scripts (`smri-eval-linear`,
`smri-eval-probe`) and as modules (`python -m eval.main_linear`). All examples use
`uv run` so the project environment is picked up.

**Linear probe** — positional args are `<model> <representation> <dataset>`:

```bash
uv run python -m eval.main_linear smri_mae cls adni_sex \
    --overrides model_kwargs.ckpt_path=/path/to/checkpoint-last.pth
```

**Trained probe** — positional args are `<model> <representation> <classifier> <dataset>`:

```bash
uv run python -m eval.main_probe smri_mae cls linear adni_ad_cn \
    --overrides model_kwargs.ckpt_path=/path/to/checkpoint-last.pth
```

`model_kwargs.ckpt_path` has no default and **must** be supplied — `smri_mae` loads
the backbone weights (and its `img_size`) from that checkpoint.

### Configuration

Defaults live in `config/default_{linear,probe}.yaml`. Three layers, last wins:

1. the default config,
2. `--config some.yaml` (a partial override file),
3. `--overrides key=value ...` (dotted-path, OmegaConf dotlist).

```bash
# bump batch size, point HF cache somewhere with space, train longer
uv run python -m eval.main_probe smri_mae patch attn adni_age \
    --overrides \
      model_kwargs.ckpt_path=/path/to/ckpt.pth \
      dataset_kwargs.cache_dir=/scratch/hf_cache \
      batch_size=8 epochs=40 metrics=[r2,rel_mae,rel_rmse,pearson_r] cv_metric=r2
```

Keys worth knowing:

- `img_size` — must match the checkpoint's training size; preprocessing pads/crops to it.
- `metrics` / `cv_metric` — classification uses `[acc, f1, bacc]`; regression uses
  `[r2, rel_mae, rel_rmse, pearson_r]`. `cv_metric` (probe) / the regression default
  `r2` selects the best hyperparameter on the validation split.
- `batch_size`, `num_workers`, `map_workers`, `amp` — throughput / memory knobs.
- probe-only: `epochs`, `lr`, `weight_decay`, `lr_scale_grid`, `wd_scale_grid`,
  `warmup_epochs`, `balanced_sampling`, `early_stopping`, `wandb`.

### On Slurm

Ready-made batch scripts live in `scripts/`:

```bash
sbatch scripts/eval_adni_linear.sbatch adni_sex cls   # args: <dataset> <representation>
```

