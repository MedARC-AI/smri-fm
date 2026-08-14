"""The overlay `survey.py` used to draw: one subject per task and modality, SynthSeg brain mask
against the mean-intensity mask the transform uses today.

    uv run python figure.py --device cuda   # -> figures/masks.png
"""

import argparse
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch

from SynthSeg_pytorch import SynthSegPredictor
from survey import CASES, EVAL_DIR, subjects_for, to_output_grid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    here = Path(__file__).parent
    (here / "figures").mkdir(exist_ok=True)
    predictor = SynthSegPredictor(device=args.device)

    fig, axes = plt.subplots(len(CASES), 3, figsize=(9, 3 * len(CASES)))
    for row_axes, (task, modality, image_tmpl, _) in zip(axes, CASES):
        sub = subjects_for(image_tmpl)[0]
        data, affine = to_output_grid(EVAL_DIR / image_tmpl.format(sub=sub))
        seg, *_ = predictor.segment(nib.Nifti1Image(data, affine))
        brain, thr = seg > 0, data > data.mean()

        for axis, dim in zip(row_axes, range(3)):
            index = [slice(None)] * 3
            index[dim] = data.shape[dim] // 2
            slab = tuple(index)
            axis.imshow(np.rot90(data[slab]), cmap="gray")
            axis.contour(np.rot90(thr[slab]), levels=[0.5], colors="tab:cyan", linewidths=0.5)
            axis.contour(np.rot90(brain[slab]), levels=[0.5], colors="tab:red", linewidths=0.7)
            axis.set_axis_off()
        row_axes[0].set_title(f"{task} {modality} {sub}", loc="left", fontsize=9)

    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    fig.suptitle(f"SynthSeg brain mask (red) vs data > mean (cyan), git {sha.stdout.strip()}")
    fig.tight_layout()
    fig.savefig(here / "figures/masks.png", dpi=110)


if __name__ == "__main__":
    main()
