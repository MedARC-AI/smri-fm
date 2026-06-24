# Synthetic Data Pipeline

This package orchestrates synthetic MR brain data generation, SynthSeg QC, and a
placeholder Hugging Face publishing step.

Generators are not vendored into this repository. The default backend expects a
checkout of `https://github.com/lukasugar/NV-Generate-CTMR` on the
`test_generation` branch through `generator_repo`. The optional `wavedit`
backend expects a checkout of `https://github.com/sisinflab/WaveDiT`.

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

For WaveDiT, prepare a separate checkout and environment:

```bash
git clone https://github.com/sisinflab/WaveDiT.git <desired_path>
cd <desired_path>
uv venv
uv pip install -r requirements.txt
```

Then set these paths in the pipeline YAML:

```yaml
generator_repo: <desired_path>
generator_python: "<desired_path>/.venv/bin/python"
```

## Config

Start from [`configs/synthetic_pipeline.example.yaml`](../../configs/synthetic_pipeline.example.yaml),
or create a YAML file:

```yaml
generator_backend: nv_generate_ctmr
generator_repo: /path/to/NV-Generate-CTMR
output_dir: /path/to/synthetic_run
num_images: 10
random_seed: 1234
output_size: null

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
  enabled: false
  repo_id: null
  private: true
  remote_dir: data
  allow_overwrite: false
```

Supported generator backends are `nv_generate_ctmr` and `wavedit`.

`output_size` is optional. If it is `null`, each backend uses its default size.
For NV, a configured `output_size` overrides the runtime `dim` and the pipeline
recomputes voxel spacing from the selected target FOV. For WaveDiT,
`output_size` is passed to `scripts/generate.py --save-size`; if omitted, the
pipeline uses WaveDiT's standard `[182, 218, 182]` crop.

WaveDiT is age-conditioned T1 whole-brain generation. Use compatible targets:

```yaml
generator_backend: wavedit
generator_repo: /path/to/WaveDiT
generator_python: "python"
num_images: 20
output_size: null

targets:
  conditions: [whole_brain]
  modalities: [mri_t1]
  planes: [axial]

wavedit:
  ages: [6, 18, 30, 45, 60, 75, 90]
  checkpoint_path: null
  checkpoint_repo: danesed/WaveDiT
  checkpoint_filename: WaveDiT-Base.pth
  checkpoint_revision: main
  num_flow_steps: 10
  sampler: heun
  cfg_scale: 1.0
  cfg_rescale: 0.7
  morpheus_scale: null
  device: cuda
```

For NV, `num_images` is per selected target. For WaveDiT, `num_images` is the
total number of images across all configured ages; the pipeline distributes
images across ages as evenly as possible.

Supported QC modes are `direct_synthseg` and `preprocess_then_synthseg`.
Supported QC metrics are `min` and `mean`. If `qc.threshold` is `null`, all
images with readable SynthSeg QC scores are accepted.

### QC Thresholds

SynthSeg QC scores are numeric values where higher is better. The pipeline
accepts an image when the selected metric is greater than or equal to
`qc.threshold`.

SynthSeg writes per-structure QC scores, not a single aggregate score. The
pipeline reduces those per-structure scores in one of two ways:

| `qc.metric` | Meaning | Behavior |
| --- | --- | --- |
| `min` | Lowest per-structure QC score for the image | Strict: rejects an image if any tracked structure has a low score |
| `mean` | Average per-structure QC score for the image | Looser: accepts images whose overall QC is good even if one structure is weaker |

Recommended workflow:

1. Run once with `threshold: null` to record QC scores without filtering.
2. Inspect `manifest.csv` columns `qc_min` and `qc_mean`.
3. Pick a threshold based on the acceptance rate you want.

Practical starting points:

| Goal | Suggested config |
| --- | --- |
| Keep most images, remove clear failures | `metric: min`, `threshold: 0.60` |
| Balanced filtering | `metric: min`, `threshold: 0.65` |
| Strict per-structure filtering | `metric: min`, `threshold: 0.70` |
| Balanced overall-quality filtering | `metric: mean`, `threshold: 0.80` |
| Strict overall-quality filtering | `metric: mean`, `threshold: 0.83` |

For the current 20-image axial T1 test run, the observed `qc_min` range was
about `0.63-0.75`, and the observed `qc_mean` range was about `0.79-0.85`.
That means `metric: min` with `threshold: 0.65` would be a reasonable first
filter, while `metric: min` with `threshold: 0.70` would be noticeably stricter.
For `metric: mean`, `threshold: 0.80` is a reasonable first filter and
`threshold: 0.83` is stricter.

### Hugging Face Publishing

Publishing runs after generation and QC, and only when `push_to_hf.enabled` is
`true`. Authentication uses the standard Hugging Face mechanisms supported by
`huggingface_hub`.

For Slurm runs, put a write-capable Hugging Face token in the repo-local `.env`
file so `scripts/synthetic_pipeline.sbatch` can export it before launching the
pipeline:

```bash
HF_TOKEN=hf_...
```

The sbatch wrapper prints which `.env` keys are set, but not their values.

| Option | Default | Meaning |
| --- | --- | --- |
| `enabled` | `false` | Run or skip the final publishing step. |
| `repo_id` | `null` | Target dataset repo, for example `username/synthetic-mri`. Required when publishing is enabled. |
| `private` | `true` | Visibility used when creating a new dataset repo. Existing repo visibility is not changed. |
| `remote_dir` | `data` | Remote prefix for QC-accepted generated images. |
| `allow_overwrite` | `false` | If `false`, publishing fails before upload when any destination file already exists. If `true`, same-path files may be replaced. |

The publisher creates the dataset repo if needed, then uploads:

```text
<remote_dir>/generated/<condition>/<modality>/<plane>/*.nii.gz
manifests/<output_dir-name>/accepted_manifest.csv
```

Existing datasets are additive as long as the new run writes to paths that do
not already exist. Use a different `remote_dir` for a separate dataset split or
run namespace, or set `allow_overwrite: true` when replacing files is intended.

## Run

Validate the config:

```bash
uv run python -m synthetic_pipeline.cli --config path/to/config.yaml --validate-only
```
uv run python -m synthetic_pipeline.cli --config src/synthetic_pipeline/configs/synthetic_pipeline.yaml --validate-only

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
sbatch scripts/synthetic_pipeline.sbatch src/synthetic_pipeline/configs/synthetic_pipeline.yaml

The script does not submit nested Slurm jobs; it runs this Python orchestrator
inside a single GPU job.
