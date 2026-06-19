from functools import lru_cache

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to


@lru_cache(maxsize=1)
def load_mni_brain_mask() -> nib.Nifti1Image:
    """The 1mm MNI152NLin2009cAsym brain mask (reoriented to RAS), loaded once."""
    import templateflow.api as tflow

    path = tflow.get(
        "MNI152NLin2009cAsym",
        resolution=1,
        desc="brain",
        suffix="mask",
        extension=".nii.gz",
    )
    return nib.as_closest_canonical(nib.load(str(path)))


def resample_binary_mask(
    mask_image: nib.Nifti1Image,
    target_image: nib.Nifti1Image,
) -> np.ndarray:
    """Resample a binary mask to an image grid with nearest-neighbor interpolation."""
    if (
        mask_image.shape == target_image.shape
        and np.allclose(mask_image.affine, target_image.affine)
    ):
        return np.asanyarray(mask_image.dataobj) > 0
    resampled = resample_from_to(mask_image, target_image, order=0)
    return np.asanyarray(resampled.dataobj) > 0
