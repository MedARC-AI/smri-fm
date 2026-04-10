# OpenNeuro

link: https://openneuro.org/

## Download

Download all raw anatomical images, JSON sidecars, and participant tables from the official OpenNeuro S3 bucket.

```bash
aws s3 sync --no-sign-request s3://openneuro.org/ data/openneuro \
  --exclude '*' \
  --include '*_T1w.*' \
  --include '*_T2w.*' \
  --include '*_FLAIR.*' \
  --include '*participants.*' \
  --exclude '*bidsignore*' \
  --exclude '*derivatives*' \
  --exclude '*desc-preproc*'
```

The data are also backed up to the MedARC R2 bucket. Download from the backup with.

```bash
aws s3 sync s3://medarc/smri-datasets/source/openneuro data/openneuro
```

## Index

We have pre-computed indexes of the images and participants:

- [`metadata/openneuro_images.csv`](metadata/openneuro_images.csv): image metadata (e.g. modality, shape, resolution, dtype)
- [`metadata/openneuro_participants.csv`](metadata/openneuro_participants.csv): subject table with age and sex

To re-compute the indexes, run
```bash
uv run scripts/index_images.py
uv run scripts/index_participants.py
```

## Curation

The [`scripts/data_curation.ipynb`](scripts/data_curation.ipynb) notebook performs a data exploration and curation. The final counts are:

|   datasets |   subjects |   images |   T1w |   T2w |   FLAIR |
|:----------:|:----------:|:--------:|:-----:|:-----:|:-------:|
|        939 |      39143 |    64287 | 51591 |  9159 |    3537 |

The curated file list is at:

- [`metadata/openneuro_include_filelist.txt`](metadata/openneuro_include_filelist.txt)

The filter criteria are:

- file size between 1MB and 60MB
- min voxel size >= 0.3mm
- X, Y (in-plane) voxel size <= 1.5mm
- Z axis voxel size <= 3mm
- X, Y, Z axis length between 120mm and 260mm

(These criteria were chosen by eye-balling the sensible outlier cutoffs for each case.)

## Notes

OpenNeuro includes several datasets that are well-known in their own right and often used individually:

- AOMIC-ID1000: [ds003097](https://openneuro.org/datasets/ds003097)
- DLBS: [ds004856](https://openneuro.org/datasets/ds004856)
- SOOP: [ds004889](https://openneuro.org/datasets/ds004889)
- QTIM: [ds004169](https://openneuro.org/datasets/ds004169)

DLBS (ds004856) and the Neurocognitive aging data release (ds003592) are good options for hold-out test sets for brain-age prediction. Both datasets have a large number of subjects spanning a wide age range, and no pathology.
