# Evaluation

This package contains the first local FOMO26 evaluation slice. It is intentionally small:
it proves the run interface, config layout, output format, and frozen-backbone probe path
before adding the rest of the challenge tasks.

## What is covered

Implemented now:

- Task 3, brain age estimation, as a scalar regression task.
- `profile: probe`, where the backbone is frozen and only a small probe is fit.
- A deterministic `dummy` backbone for smoke tests and interface development.
- Synthetic Task 3 data for local development before the real FOMO26 files are wired in.
- A ridge-regression probe head fit on frozen embeddings.
- Stable per-run outputs:
  - run-level `config.json`
  - run-level `metrics.json`
  - run-level `run_metadata.json`
  - per-task `metrics.json`
  - per-task `predictions.csv`
  - per-task `run_metadata.json`

Not implemented yet:

- Loading real FOMO26 Task 3 files.
- Tasks 1, 2, 4, 5, 6, and 7.
- `profile: full` fine-tuning.
- Container submission/validation flow.

## How to run

Run from the repository root.

```bash
uv run python -c "from evaluation import run_evals_from_config; run_evals_from_config('src/evaluation/config/default_probe.yaml')"
```

The default config writes outputs to:

```text
fomo26_runs/20260512-091530Z__probe__tasks-3__dummy/
```

Each invocation creates a new timestamped run directory, so rerunning the same config
does not overwrite earlier results.

You can also call the Python API directly:

```python
from evaluation import ProbeConfig, run_evals
from evaluation.tasks.task3_brain_age import BrainAgeTaskConfig

run_evals(
    model="dummy",
    profile="probe",
    tasks=["3"],
    output_dir="fomo26_runs",
    model_kwargs={"embedding_dim": 8, "input_shape": [1, 8, 8, 8]},
    task_configs={
        "3": BrainAgeTaskConfig(
            source="synthetic",
            n_train=32,
            n_validation=16,
            noise_std=0.1,
        )
    },
    probe_config=ProbeConfig(head="ridge", alpha=1.0),
)
```

Run the evaluation tests with:

```bash
uv run pytest tests/test_evaluation_probe.py
```

## Config

The default config lives at `src/evaluation/config/default_probe.yaml`.

The layout follows the same broad idea as Brainmarks: keep run settings in a YAML file,
load it through OmegaConf, and pass named sections to registries and task runners. The
main difference is that this package converts those sections into small dataclass config
objects at the API boundary. That keeps the YAML easy to edit while making task code more
explicit than plain `dict` arguments.

Current fields:

- `model`: model registry key or model object passed through the Python API. The only
  registered value today is `dummy`.
- `profile`: evaluation profile. `probe` is implemented and freezes the backbone before
  extracting embeddings. `full` is reserved for future fine-tuning support.
- `tasks`: list of task ids to run. Currently only `"3"` is implemented.
- `output_dir`: root directory for timestamped run outputs. Defaults to `fomo26_runs`.
- `data_dir`: root directory for local task data. Present now for the future real-data
  loaders; the synthetic Task 3 loader does not read from it.
- `name`: optional human-readable run label. If provided, it is sanitized and inserted
  into the generated run id, for example
  `20260512-091530Z__debug-run__probe__tasks-3__dummy`.
- `seed`: random seed used by task data generation and future split logic.
- `device`: torch device used for backbone embedding extraction, for example `cpu` or
  `cuda`.
- `model_kwargs`: keyword arguments for the selected model factory. For `dummy`, these
  include:
  - `embedding_dim`: number of embedding features produced by the dummy backbone.
  - `input_shape`: image tensor shape excluding batch dimension.
- `task_configs`: mapping from task id to task-specific config. For Task 3:
  - `source`: data source. Only `synthetic` is implemented today.
  - `n_train`: number of synthetic training samples.
  - `n_validation`: number of synthetic validation samples.
  - `noise_std`: Gaussian label noise for synthetic ages.
- `probe_config`: shared frozen-backbone probe settings:
  - `head`: probe head type. Task 3 currently supports `ridge`.
  - `alpha`: ridge regularization strength.
  - `kwargs`: optional extra keyword arguments passed to the probe implementation.

## Design decisions

The implementation is deliberately registry-based. Models are created through
`evaluation.models.registry`, and tasks are created through `evaluation.tasks.registry`.
That keeps the public API stable as more tasks and backbones are added.

The config is split into three levels:

- `RunConfig` describes the whole evaluation run.
- `ProbeConfig` describes the shared frozen-backbone probe settings.
- each task can define its own `TaskConfig` subclass, such as `BrainAgeTaskConfig`.

This avoids a single large config object that must know every task-specific field in
advance. Future segmentation, classification, embedding, and fairness tasks can each
declare their own config fields while still sharing the same run interface.

The output directory is split into run-level and task-level artifacts:

```text
fomo26_runs/
  20260512-091530Z__probe__tasks-3-4__dummy/
    config.json
    metrics.json
    run_metadata.json
    task_3_brain_age/
      metrics.json
      predictions.csv
      run_metadata.json
    task_4_<name>/
      metrics.json
      run_metadata.json
```

The timestamp is UTC and comes first so runs sort chronologically. If two runs start in
the same second, the later directory receives a suffix such as `__01`.

The first implemented task uses synthetic data because the immediate goal is the interface
and probe mechanics, not an official local score. The synthetic loader is intentionally
kept behind `source: synthetic`, so adding real-data loading later does not change the
public API.

The backbone is frozen in `run_evals` before any task runs. Task metadata records
`backbone_trainable_parameters`, which should be `0` in probe mode.

## Extending with new tasks

To add a task:

1. Create a new module in `src/evaluation/tasks/`, for example `task5_pmg.py`.
2. Define a task-specific config dataclass that subclasses `TaskConfig`.
3. Define an `EvalTask` subclass with:
   - `id`
   - `name`
   - `output_name`
   - `config_type`
   - `run_probe(...)`
4. In `run_probe`, load or construct the task data, extract frozen embeddings with the
   provided backbone, fit the appropriate probe head, compute official/local metrics, and
   write stable per-task outputs.
5. Register the task id in `src/evaluation/tasks/registry.py`.
6. Add a `task_configs` entry to `src/evaluation/config/default_probe.yaml` or a new
   task-specific config file.
7. Add focused tests that check:
   - the task can be selected through `run_evals`
   - expected files are written
   - metrics have the expected keys
   - probe mode keeps the backbone frozen

For classification tasks, the first reusable addition should probably be a shared
classification probe helper rather than duplicating AUROC/F1 logic inside each task. For
segmentation tasks, keep dense prediction and geometry handling task-specific until the
common NIfTI/mask contracts are clear.

## Next steps

- Replace Task 3 synthetic data with a real local FOMO26 loader while preserving
  `BrainAgeTaskConfig`.
- Add Task 5 next, because it is a single-modality binary classification task and should
  exercise a reusable classification probe path.
- Add a shared metrics module for regression and classification metrics.
- Add real model wrappers that expose the same `encode(batch)` contract as the dummy
  backbone.
- Add CLI entry points once the Python API and config surface settle.
- Add `profile: full` only after fine-tuning is implemented as a standalone feature.
