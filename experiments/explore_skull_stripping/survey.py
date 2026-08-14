"""Should the eval transform skull-strip with SynthSeg, and where?

Pretraining masked every volume with SynthSeg; `SmriMaeTransform` uses `data > data.mean()`
instead, which keeps skull, scalp and neck. This runs the SynthSeg pytorch port on the grid the
transform produces and, per task and modality, compares three candidate masks:

  thr    data > data.mean()   what the transform does today
  pos    data > 0             free, and exact wherever the challenge already stripped the volume
  brain  SynthSeg seg > 0     what pretraining did

reporting voxel counts, the live-token counts those masks induce in the encoder, and -- where
the task ships a voxelwise label -- the fraction of the label each mask keeps. A mask that
erases the target is disqualifying whatever it does for fidelity: outside the mask the
transform writes zeros, so the structure is gone before the backbone sees it.

Roughly 24s per volume wherever it runs -- SynthSeg's numpy preprocessing and its connected
component postprocessing dominate, not the network -- so it is sharded by task and the shards run
in parallel. `figure.py` draws the overlay separately.

    for t in task1 task2 task3 task4 task5; do uv run python survey.py --device cuda --tasks $t & done
    uv run python -c "import glob,pandas; pandas.concat(map(pandas.read_csv, glob.glob('survey_task*.tsv')), ...)"
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from fomo_tune.backbone import fit_to_shape, rescale
from SynthSeg_pytorch import SynthSegPredictor

ROOT = Path(__file__).parents[2]
EVAL_DIR = ROOT / "data/fomo_eval"
IMG_SIZE = (208, 240, 208)
PATCH = 8
N_SUBJECTS = 6

# task, modality, image template, label template (None where the task has no voxelwise label)
CASES = [
    (
        "task1",
        "adc",
        "Task_1/preprocessed/{sub}/ses-01/adc.nii.gz",
        "Task_1/labels/{sub}/ses-01/seg.nii.gz",
    ),
    (
        "task1",
        "dwi_b1000",
        "Task_1/preprocessed/{sub}/ses-01/dwi_b1000.nii.gz",
        "Task_1/labels/{sub}/ses-01/seg.nii.gz",
    ),
    (
        "task1",
        "flair",
        "Task_1/preprocessed/{sub}/ses-01/flair.nii.gz",
        "Task_1/labels/{sub}/ses-01/seg.nii.gz",
    ),
    (
        "task2",
        "dwi_b1000",
        "Task_2/preprocessed/{sub}/ses-01/dwi_b1000.nii.gz",
        "Task_2/labels/{sub}/ses-01/seg.nii.gz",
    ),
    (
        "task2",
        "flair",
        "Task_2/preprocessed/{sub}/ses-01/flair.nii.gz",
        "Task_2/labels/{sub}/ses-01/seg.nii.gz",
    ),
    ("task3", "t1w", "Task_3/preprocessed/{sub}/ses-01/t1w.nii.gz", None),
    (
        "task4",
        "t2w",
        "Task_4/preprocessed/{sub}/ses-01/t2w.nii.gz",
        "Task_4/labels/{sub}/ses-01/seg.nii.gz",
    ),
    ("task5", "t1", "Task_5/preprocessed/{sub}/ses_01/t1.nii.gz", None),
]


def to_output_grid(path: Path, nearest: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """The geometry half of `SmriMaeTransform`: canonical RAS, 1mm, centred in IMG_SIZE.

    `nearest` is for labels. `nearest-exact`, not `nearest`: it samples at the same
    `(j + 0.5) / scale - 0.5` the transform's trilinear resample uses, so a label stays aligned
    with its image. Plain `nearest` would shift it by up to half an input voxel, which is 3mm on
    the 6.5mm slices in tasks 1 and 2.
    """
    img = nib.as_closest_canonical(nib.funcs.squeeze_image(nib.load(path)))
    data = torch.from_numpy(np.ascontiguousarray(img.get_fdata(dtype=np.float32)))
    affine = np.asarray(img.affine)
    spacing = img.header.get_zooms()
    if max(abs(s - 1.0) for s in spacing) > 0.05:
        if nearest:
            scales = tuple(float(s) for s in spacing)
            resampled = F.interpolate(data[None, None], scale_factor=scales, mode="nearest-exact")
            step = np.diag([*(1 / np.asarray(scales)), 1.0])
            step[:3, 3] = 0.5 / np.asarray(scales) - 0.5
            data, affine = resampled.squeeze(0, 1), affine @ step
        else:
            data, affine = rescale(data, affine, spacing, (1.0, 1.0, 1.0))
    data, affine = fit_to_shape(data, affine, target_shape=IMG_SIZE)
    return data.numpy(), affine


def live_tokens(mask: np.ndarray) -> int:
    """Patches the encoder keeps: `patch_num_obs > 0`, so one voxel validates a whole patch."""
    grid = (IMG_SIZE[0] // PATCH, PATCH, IMG_SIZE[1] // PATCH, PATCH, IMG_SIZE[2] // PATCH, PATCH)
    return int(mask.reshape(grid).any(axis=(1, 3, 5)).sum())


def subjects_for(image_tmpl: str) -> list[str]:
    task_dir = EVAL_DIR / image_tmpl.split("/")[0]
    pattern = "/".join(image_tmpl.split("/")[1:]).format(sub="*")
    return sorted(p.parts[-3] for p in task_dir.glob(pattern))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-subjects", type=int, default=N_SUBJECTS)
    parser.add_argument("--tasks", default="task1,task2,task3,task4,task5")
    args = parser.parse_args()

    here = Path(__file__).parent
    wanted = set(args.tasks.split(","))
    predictor = SynthSegPredictor(device=args.device)

    rows = []
    for task, modality, image_tmpl, label_tmpl in CASES:
        if task not in wanted:
            continue
        subs = subjects_for(image_tmpl)[: args.n_subjects]
        for sub in subs:
            data, affine = to_output_grid(EVAL_DIR / image_tmpl.format(sub=sub))
            seg, *_ = predictor.segment(nib.Nifti1Image(data, affine))

            masks = {"thr": data > data.mean(), "pos": data > 0, "brain": seg > 0}
            row = {"task": task, "modality": modality, "subject": sub}
            for name, mask in masks.items():
                row[f"{name}_kvox"] = mask.sum() / 1e3
                row[f"{name}_tokens"] = live_tokens(mask)

            label_path = None if label_tmpl is None else EVAL_DIR / label_tmpl.format(sub=sub)
            if label_path is not None and label_path.exists():
                label, _ = to_output_grid(label_path, nearest=True)
                label = label > 0
                row["label_kvox"] = label.sum() / 1e3
                for name, mask in masks.items():
                    row[f"{name}_label_kept"] = (label & mask).sum() / label.sum()

            # what SynthSeg adds to, and drops from, an already-stripped support
            row["brain_outside_pos"] = (masks["brain"] & ~masks["pos"]).sum() / masks["brain"].sum()
            row["pos_dropped"] = (masks["pos"] & ~masks["brain"]).sum() / masks["pos"].sum()
            rows.append(row)
            print(row, flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(here / f"survey_{args.tasks}.tsv", sep="\t", index=False, float_format="%.4f")


if __name__ == "__main__":
    main()
