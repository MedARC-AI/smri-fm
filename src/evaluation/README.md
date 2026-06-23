# Evaluation

Internal evaluation suite. Currently only supporting frozen-feature sklearn linear probe.

## Run

```bash
uv run python -m evaluation.main_linear <model> <task> [--config cfg.yaml] [--overrides key=value ...]
# e.g.
uv run python -m evaluation.main_linear smri_mae dlbs_age --overrides model_kwargs.ckpt_path=/path/to/ckpt.pth
```

`model` and `task` are registered names (the CLI `--help` lists them). Run-level
settings come from [config/default_linear.yaml](config/default_linear.yaml),
overridden by an optional `--config` and then dot-list `--overrides`.

Outputs save in `<output_root>/<name>/` (default name `<model>__<task>`):

- `summary.csv`: one row of `model, task, tput, <metric>, <metric>_std`
- `metrics.json`: the summary, per-fold scores, primary metrics, and whether the
  task is eligible for model selection
- `config.yaml`: the fully resolved config
- `log.txt`: run log

Task-specific metric policy:

- Sex is a canary only: balanced accuracy and AUROC, with probe regularization
  selected by inner CV.
- AD vs CN uses AUROC as primary, with AUPRC and balanced accuracy.
- Brain age uses Pearson r and R² as primary. MAE is reported in years after
  age-bias correction fitted on each training fold.
- SynthSeg volumes report R² and Pearson correlation for every region, plus
  macro means.
- BAG applies the same age-bias correction and age metrics before computing its
  case-control t-statistic.

### Age+sex floor (`covariate_columns`)

Clinical targets correlate with age, and the image latent encodes age, so a probe
can post a real score by riding the age axis without carrying target-specific
signal. To separate the two, a `ColumnTask` may set `covariate_columns` (e.g.
`("age", "sex")`). On each fold the harness then also fits, on the *same* split:

- **floor** (A) — a covariate-only model on `[age, age², sex]` (the quadratic age
  term lets the floor model the curved age-vs-biomarker relationship);
- **combined** (B) — a model on `[latent, age, age², sex]`.

For every metric `<m>` it reports `floor_<m>`, `combined_<m>`, and
`gap_<m> = combined_<m> − floor_<m>` alongside the latent-only headline. The gap is
the latent's contribution *beyond* age/sex. Sign convention: `gap = combined − floor`
for all metrics, so a positive gap means the latent helped for higher-is-better
metrics (AUROC, R²) and a negative gap means it helped for error metrics (MAE).
Canaries (`adni_age`, `adni_sex`) leave `covariate_columns` empty — the floor is
degenerate there — and tasks without it emit only the headline metrics.

## Architecture

- [main_linear.py](main_linear.py) is the main entrypoint
- [models/](models/) contains model wrappers, e.g. [models/smri_mae.py](models/smri_mae.py). Each model defines a transform (`nib.Nifti1Image -> sample dict`) as well as the model itself (`batch dict -> embeddings`).
- [tasks/](tasks/) contains defined tasks, e.g. [tasks/fomo.py](tasks/fomo.py). Each task consists of a dataset as well as defined targets, splits, and scoring metrics.

## Adding things

Tasks and models share a registry: a builder decorated with `@register_task` /
`@register_model`, discovered automatically and constructed by name.

- **Task**: implement the `Task` protocol and decorate a builder with
  `@register_task`. For predicting a column of an HF dataset, use `ColumnTask`
  ([tasks/column.py](tasks/column.py)) with a sklearn splitter. See
  [tasks/dlbs.py](tasks/dlbs.py) and [tasks/fomo.py](tasks/fomo.py).
- **Model**: write a `(Model, Transform)` pair and decorate the builder with
  `@register_model`. See [models/smri_mae.py](models/smri_mae.py).
- **Dataset**: add a reproducible builder returning an HF `Dataset` of niftis +
  metadata next to its task (see `load_dlbs_t1w` / `load_fomo_task3`).
