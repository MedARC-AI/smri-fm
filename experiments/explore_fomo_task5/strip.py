"""Brain-masked copies of the T1s, for loading into a viewer.

SynthSeg's label map is the mask. `seg > 0` keeps no skull but does keep label 24,
extracerebral CSF, which is 18-22% of it here and sits outside the pial surface, so the mask
is `seg > 0` with label 24 dropped.

Writes `t1_1mm_brain.nii.gz` beside each subject's segmentation, or `t1_native_brain.nii.gz`
under `--native`, which strips the original nifti instead. Native is 0.43-0.77mm in plane on the
coronal acquisition, so it is the one to open if the question is about cortex.

    uv run python experiments/explore_fomo_task5/strip.py
    uv run python experiments/explore_fomo_task5/strip.py --native
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np

from montage import on_grid
from segment import TASK_DIR, subjects

# SynthSeg's extracerebral CSF class, the subarachnoid space outside the pial surface.
CSF = 24


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "output")
    parser.add_argument("--native", action="store_true", help="strip the original grid")
    parser.add_argument("--subjects", default="", help="comma-separated subset, e.g. sub_45")
    args = parser.parse_args()
    wanted = set(args.subjects.split(",")) if args.subjects else None

    for sub, label in subjects():
        if wanted is not None and sub not in wanted:
            continue
        seg_img = nib.load(args.out / sub / "seg.nii.gz")
        if args.native:
            image = nib.as_closest_canonical(
                nib.load(TASK_DIR / f"preprocessed/{sub}/ses_01/t1.nii.gz")
            )
            labels = on_grid(seg_img, image)
            name = "t1_native_brain.nii.gz"
        else:
            image = nib.load(args.out / sub / "t1_1mm.nii.gz")
            labels = np.asarray(seg_img.dataobj)
            name = "t1_1mm_brain.nii.gz"

        brain = (labels > 0) & (labels != CSF)
        data = np.where(brain, np.asarray(image.dataobj, dtype=np.float32), 0.0)
        stripped = nib.Nifti1Image(data, image.affine, image.header)
        stripped.set_data_dtype(np.float32)
        nib.save(stripped, args.out / sub / name)
        print(f"{sub} label={label} {data.shape} kept {brain.mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
