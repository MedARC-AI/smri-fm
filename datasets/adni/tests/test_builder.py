from pathlib import Path

import nibabel as nib
import numpy as np
from datasets import Dataset

from adni_curate import curate
adni_features = curate.adni_features


def test_generator_yields_nifti_bytes(tmp_path: Path):
    image_path = tmp_path / "sample.nii.gz"
    mask_path = tmp_path / "sample_mask.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((3, 4, 5), dtype=np.float32), np.eye(4)), image_path)
    nib.save(nib.Nifti1Image(np.ones((3, 4, 5), dtype=np.uint8), np.eye(4)), mask_path)
    record = {
        "sample_id": "sample", "participant_id": "subject", "session_id": "20200101",
        "scan_date": "2020-01-01", "age": 70.0, "sex": "Female", "diagnosis": "AD",
        "synthseg_volumes": [1.0] * 101, "synthseg_qc_mean": 0.9,
        "synthseg_qc_min": 0.8, "train_rank": 0, "path": "images/sample.nii.gz",
        "local_path": str(image_path), "mask_path": "masks/sample_mask.nii.gz",
        "local_mask_path": str(mask_path),
    }
    dataset = Dataset.from_generator(
        curate._generate_samples,
        features=adni_features(),
        gen_kwargs={"records": [record]},
    )

    assert dataset[0]["nifti"].shape == (3, 4, 5)
    assert dataset[0]["brain_mask"].shape == (3, 4, 5)
    assert dataset[0]["path"] == "images/sample.nii.gz"
    assert dataset[0]["mask_path"] == "masks/sample_mask.nii.gz"
    assert dataset[0]["diagnosis"] == 1
    assert "synthseg_qc_mean" not in dataset.column_names
    assert "synthseg_qc_min" not in dataset.column_names
