"""SynthSeg every task-5 volume, keeping the segmentation this time.

Fed the native nifti rather than the eval transform's 208x240x208 canvas: SynthSeg resamples to
1mm itself, so this returns anatomy on a 1mm RAS grid at the scan's own field of view, which is
what a cortical surface wants. `t1_1mm.nii.gz` is the resampled input on that same grid.

    uv run python experiments/explore_fomo_task5/segment.py --device cuda \
        | tee experiments/explore_fomo_task5/output/segment.log            # ~30s/subject
"""

import argparse
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from SynthSeg_pytorch import SynthSegPredictor

ROOT = Path(__file__).parents[2]
TASK_DIR = ROOT / "data/fomo_eval/Task_5"


def subjects() -> list[tuple[str, int]]:
    rows = []
    for path in sorted((TASK_DIR / "preprocessed").iterdir()):
        label = (TASK_DIR / "labels" / path.name / "ses_01/labels.txt").read_text().strip()
        rows.append((path.name, int(label)))
    return rows


def scanner(sub: str) -> str:
    """Which of the cohort's two scanners the volume was built for.

    `Task_5_extract.py` resizes every JPEG slice to the chosen scanner's native reconstructed
    matrix, 260x320 for the 3T Skyra and 512x512 for the 1.5T Cigna GE, so the in-plane matrix
    of the nifti records the choice exactly. It records a decision rather than measuring one:
    for cases the scanner came from the dataset authors by correspondence, and for controls and
    unlisted cases from the raw JPEG aspect ratio.
    """
    shape = nib.load(TASK_DIR / "preprocessed" / sub / "ses_01/t1.nii.gz").shape
    return "Skyra" if shape[0] == 260 else "Cigna"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "output")
    parser.add_argument("--subjects", default="", help="comma-separated subset, e.g. sub_01,sub_25")
    args = parser.parse_args()

    wanted = set(args.subjects.split(",")) if args.subjects else None
    predictor = SynthSegPredictor(device=args.device)

    for sub, label in subjects():
        if wanted is not None and sub not in wanted:
            continue
        sub_dir = args.out / sub
        sub_dir.mkdir(parents=True, exist_ok=True)

        start = time.perf_counter()
        seg, _, volumes, aff, header = predictor.segment(
            TASK_DIR / "preprocessed" / sub / "ses_01/t1.nii.gz",
            path_resample=str(sub_dir / "t1_1mm.nii.gz"),
        )
        out = nib.Nifti1Image(seg.astype(np.uint8), aff, header)
        out.set_data_dtype(np.uint8)
        nib.save(out, sub_dir / "seg.nii.gz")

        print(
            f"{sub} label={label} {seg.shape} "
            f"cortex_ml={(np.isin(seg, (3, 42)).sum() / 1e3):.1f} "
            f"total_ml={(volumes[0] / 1e3):.1f} ({time.perf_counter() - start:.0f}s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
