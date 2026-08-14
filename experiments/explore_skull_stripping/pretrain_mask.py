"""How close is the port's default path to the mask pretraining actually used?

Pretraining ran SynthSeg with `--parc --robust` -- the three-network SynthSeg+ path -- and took
`seg > 0`. `SynthSeg_pytorch` ports only the default single-network path; `--robust` would mean
the 215MB checkpoint and porting a denoiser and a second network. `data/PT001_ClevelandCCF/` is
one FOMO300 subset copied whole, so it carries the pipeline's own masks on the same 1mm grid and
the comparison is exact -- no resampling, no registration in between.

    uv run python pretrain_mask.py --device cuda
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch

from SynthSeg_pytorch import SynthSegPredictor

ROOT = Path(__file__).parents[2]
SUBSET = ROOT / "data/PT001_ClevelandCCF"
SPACE = "space-MNI152NLin2009cAsym"
N_SUBJECTS = 8


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-subjects", type=int, default=N_SUBJECTS)
    args = parser.parse_args()

    predictor = SynthSegPredictor(device=args.device)
    images = sorted((SUBSET / "processed").glob(f"*_{SPACE}_desc-processed.nii.gz"))

    rows = []
    for image_path in images[: args.n_subjects]:
        stem = image_path.name.removesuffix(f"_{SPACE}_desc-processed.nii.gz")
        reference_path = SUBSET / f"derivatives/masks/{stem}_{SPACE}_desc-brain_mask.nii.gz"
        stored_seg_path = SUBSET / f"derivatives/synthseg/{stem}_desc-synthseg_dseg.nii.gz"

        seg, *_ = predictor.segment(image_path)
        port = seg > 0
        reference = np.asanyarray(nib.load(reference_path).dataobj) > 0
        stored = np.asanyarray(nib.load(stored_seg_path).dataobj) > 0
        assert port.shape == reference.shape == stored.shape

        rows.append(
            {
                "subject": stem,
                "port_kvox": port.sum() / 1e3,
                "reference_kvox": reference.sum() / 1e3,
                "dice_port_reference": 2
                * (port & reference).sum()
                / (port.sum() + reference.sum()),
                "dice_stored_reference": 2
                * (stored & reference).sum()
                / (stored.sum() + reference.sum()),
                "port_extra": (port & ~reference).sum() / reference.sum(),
                "port_missed": (reference & ~port).sum() / reference.sum(),
            }
        )
        print(rows[-1], flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(
        Path(__file__).parent / "pretrain_mask.tsv", sep="\t", index=False, float_format="%.4f"
    )
    print(table.drop(columns="subject").mean())


if __name__ == "__main__":
    main()
