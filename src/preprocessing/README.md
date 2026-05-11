# MRI Preprocessing

Processes anatomical images into MNI-space outputs. Supports T1w, T2w, and FLAIR inputs.

## Pipeline

1. Rigid registration to TemplateFlow `MNI152NLin2009cAsym` (ANTs)
2. Run SynthSeg on the processed image
3. Save a binary brain mask from the SynthSeg segmentation (`dseg > 0`)

## Outputs

For each input file the pipeline writes:

| File | Description |
|---|---|
| `<input>/processed/*_space-MNI152NLin2009cAsym_desc-processed.nii.gz` | MNI-space processed image |
| `<input>/derivatives/masks/*_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz` | Binary brain mask for skull stripping |
| `<input>/derivatives/transforms/*_from-native_to-MNI152NLin2009cAsym_mode-image_xfm.mat` | Rigid transform to MNI |

Brains smaller than the template field of view will be surrounded by zeros. Anatomy that falls outside the template boundary after rigid alignment is clipped.

## SynthSeg Derivatives

The full pipeline also writes SynthSeg derivatives under `<input>/derivatives/synthseg/`:

| File | Description |
|---|---|
| `*_desc-synthseg_dseg.nii.gz` | Tissue segmentation |
| `*_volumes.csv` | Raw SynthSeg volumetric measurements |
| `*_qc.csv` | Raw SynthSeg QC scores |

## Local Runtime

Install the unified environment from the repo root:

```bash
uv sync
```

The pipeline scans `<input>` recursively for `.nii.gz` files and writes
outputs alongside them:

- processed outputs: `<input>/processed/`
- logs: `<input>/logs/`
- derivatives: `<input>/derivatives/`

Run the full pipeline:

```bash
uv run python src/preprocessing/pipeline.py --input <input>
```

Or submit as a Slurm job:

```bash
sbatch scripts/preprocess.sbatch <input>
```

The default SynthSeg backend is:

```bash
uvx --python 3.11 --from 'git+https://github.com/MedARC-AI/SynthSeg.git' SynthSeg
```
