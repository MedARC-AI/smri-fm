# MRI Preprocessing

Preprocesses BIDS anatomical images and writes MNI-space derivatives. Supports T1w, T2w, and FLAIR inputs.

## Pipeline

1. Reorient to RAS
2. Skull stripping with SynthStrip (FreeSurfer 7.4.1)
3. Resample to 1 mm isotropic when needed (ANTs B-spline interpolation)
4. Rigid registration to TemplateFlow `MNI152NLin2009cAsym` (ANTs)

## Outputs

For each input file the pipeline writes three derivatives (BIDS-compliant naming):

| File | Description |
|---|---|
| `*_space-MNI152NLin2009cAsym_desc-preproc_{suffix}.nii.gz` | MNI-space preprocessed image |
| `*_space-MNI152NLin2009cAsym_desc-brain_mask_{suffix}.nii.gz` | MNI-space brain mask |
| `*_from-native_to-MNI152NLin2009cAsym_mode-image_desc-{suffix}_xfm.mat` | Rigid transform to MNI |

Brains smaller than the template field of view will be surrounded by zeros. Anatomy that falls outside the template boundary after rigid alignment is clipped.

## Optional: SynthSeg

Pass `--synthseg` to run FreeSurfer's `mri_synthseg` on preprocessed outputs, producing:

| File | Description |
|---|---|
| `*_res-1mm_desc-synthseg_dseg.nii.gz` | Tissue segmentation |
| `*_desc-synthseg_dseg.tsv` | Label definitions (51 tissue classes) |
| `*_desc-synthseg_volumes.tsv` | Volumetric measurements (GMV, WMV, sGMV, TCV, DK parcellation) |
| `*_desc-synthseg_qc.tsv` | QC scores |

## Docker

Build from the repo root (build context must be the repo root, not this directory):

```bash
docker build -f preprocessing/Dockerfile -t smri-fm-preproc .
```

Run against a BIDS directory:

```bash
docker run --rm --gpus all \
  -v /path/to/bids:/app/preprocessing/data/raw \
  -v /path/to/output:/app/preprocessing/data/processed \
  -v /path/to/logs:/app/preprocessing/data/logs \
  smri-fm-preproc --n_workers 48
```

With SynthSeg and a custom derivatives directory:

```bash
docker run --rm --gpus all \
  -v /path/to/bids:/app/preprocessing/data/raw \
  -v /path/to/output:/app/preprocessing/data/processed \
  -v /path/to/logs:/app/preprocessing/data/logs \
  -v /path/to/derivatives:/app/preprocessing/data/derivatives \
  smri-fm-preproc --n_workers 48 --synthseg
```

