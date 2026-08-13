# MedARC structural MRI foundation model

<a href="https://discord.gg/tVR4TWnRM9"><img src="https://img.shields.io/badge/Discord-Collaborate%20with%20us-5865F2?style=for-the-badge&logo=discord&logoColor=white" /></a>

Open research project building a structural MRI foundation model, targeting
the [FOMO26](https://fomo26.github.io/) challenge.

For the final FOMO26 push, we are working exclusively in [`src/fomo_tune/`](src/fomo_tune/).

## Installation

```bash
git clone -b fomo_tune --recurse-submodules https://github.com/MedARC-AI/smri-fm fomo_tune
cd fomo_tune
uv sync
uv run pre-commit install --install-hooks
```

If you are on the medarc cluster, use the shared huggingface cache

```bash
export HF_HOME="/data/smri-datasets/huggingface"
```

This will avoid re-downloading the model checkpoints and datasets.

### Quickstart

Train the default probe head for Task 1.

```bash
uv run python -m fomo_tune.main_task1 train
```

## Layout

| Path | |
|---|---|
| [`src/fomo_tune/`](src/fomo_tune/) | The active work. One `main_task<k>.py` per task, plus the frozen backbone, data loading and container build. See the [`README.md`](src/fomo_tune/README.md) for more info. |
| [`src/smri_mae/`](src/smri_mae/) | MAE pretraining code. Currently stable, not accepting changes. |
| `experiments/<name>/` | Self-contained experiments: config, launch script, analysis, figures. |
| `third_party/` | Third party code. Has the challenge's container validator, and its submission spec. |
| `CODING_STANDARDS.md` | Read before writing code. |
| `AGENTS.md` | Orientation specifically for coding agents. |
