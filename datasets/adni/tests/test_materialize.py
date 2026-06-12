from pathlib import Path

import nibabel as nib
import numpy as np

from adni_curate import curate

_generate_samples = curate._generate_samples


def test_generator_preserves_nifti_path(tmp_path: Path):
    image = tmp_path / "sample.nii.gz"
    mask = tmp_path / "sample_mask.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((2, 3, 4), dtype=np.float32), np.eye(4)), image)
    nib.save(nib.Nifti1Image(np.ones((2, 3, 4), dtype=np.uint8), np.eye(4)), mask)
    record = {
        "sample_id": "sample", "participant_id": "subject", "session_id": "20200101",
        "scan_date": "2020-01-01", "age": 70.0, "sex": "Female", "diagnosis": "AD",
        "synthseg_volumes": [1.0] * 101, "synthseg_qc_mean": 0.9,
        "synthseg_qc_min": 0.8, "train_rank": 0, "path": "images/sample.nii.gz",
        "local_path": str(image), "mask_path": "masks/sample_mask.nii.gz",
        "local_mask_path": str(mask),
    }

    sample = next(_generate_samples([record]))

    assert sample["nifti"]["path"] == "images/sample.nii.gz"
    assert sample["nifti"]["bytes"] == image.read_bytes()
    assert sample["brain_mask"]["path"] == "masks/sample_mask.nii.gz"
    assert sample["brain_mask"]["bytes"] == mask.read_bytes()
    assert sample["path"] == "images/sample.nii.gz"
    assert sample["mask_path"] == "masks/sample_mask.nii.gz"
