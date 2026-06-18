# Synthetic Data Pipeline

This package orchestrates synthetic MR brain data generation, SynthSeg QC, and a
placeholder Hugging Face publishing step.

The generator is not vendored into this repository. Provide a checkout of
`https://github.com/lukasugar/NV-Generate-CTMR` on the `test_generation` branch
through `generator_repo`.

## Setup

Install this repository environment from the repo root:

```bash
uv sync
```

Prepare the generator checkout separately:

```bash
git clone --branch test_generation https://github.com/lukasugar/NV-Generate-CTMR.git
cd NV-Generate-CTMR
uv venv
uv pip install -r requirements.txt
```

The pipeline runs generator commands with `uv run --frozen python` from inside
that checkout by default. If your generator environment is different, set
`generator_python` in the YAML config, for example `generator_python: "python"`.

## Config

Start from [`configs/synthetic_pipeline.example.yaml`](../../configs/synthetic_pipeline.example.yaml),
or create a YAML file:

```yaml
generator_repo: /path/to/NV-Generate-CTMR
output_dir: /path/to/synthetic_run
num_images: 10
random_seed: 1234

targets:
  conditions: [whole_brain]
  modalities: [mri_t1, mri_t2, mri_flair]
  planes: [axial]

qc:
  mode: direct_synthseg
  threshold: null
  metric: min
  synthseg_cmd: "uvx --python 3.11 --from 'git+https://github.com/MedARC-AI/SynthSeg.git' SynthSeg"
  threads: 8
  cpu: false

push_to_hf:
  enabled: true
```

Supported QC modes are `direct_synthseg` and `preprocess_then_synthseg`.
Supported QC metrics are `min` and `mean`. If `qc.threshold` is `null`, all
images with readable SynthSeg QC scores are accepted.

## Run

Validate the config:

```bash
uv run python -m synthetic_pipeline.cli --config path/to/config.yaml --validate-only
```

Run the full pipeline:

```bash
uv run python -m synthetic_pipeline.cli --config path/to/config.yaml
```

Useful overrides:

```bash
uv run python -m synthetic_pipeline.cli \
  --config path/to/config.yaml \
  --num-images 2 \
  --qc-threshold 0.65 \
  --qc-metric mean
```

## Outputs

The pipeline writes:

```text
<output_dir>/
  generated/<condition>/<modality>/<plane>/*.nii.gz
  runtime_configs/
  derivatives/synthseg/
  logs/
  manifest.csv
  accepted_manifest.csv
```

`manifest.csv` keeps every generated image and its QC status.
`accepted_manifest.csv` contains only rows that passed the configured QC rule.

## Slurm

Submit with:

```bash
sbatch scripts/synthetic_pipeline.sbatch path/to/config.yaml [extra CLI flags]
```

The script does not submit nested Slurm jobs; it runs this Python orchestrator
inside a single GPU job.
