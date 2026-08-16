"""Fit the current Task 3 baseline on SALD and report external DLBS metrics."""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from fomo_tune.datasets import load_fomo_task3
from fomo_tune.main_task3 import Config, Task3Method

DEFAULT_CKPT = (
    "hf://medarc/walnut/checkpoints/walnut-v0-1/vitl/sub-52k/checkpoint-last.pth"
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("output/task3_dlbs_external"))
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    method = Task3Method(Config(ckpt_path=args.ckpt))
    method.fit(list(load_fomo_task3()))

    predictions = pd.read_csv(args.eval_manifest, sep="\t")
    predictions["prediction"] = [
        method.predict({"t1w": nib.load(path)}) for path in predictions.path
    ]
    predictions["absolute_error"] = np.abs(predictions.prediction - predictions.age)
    predictions.to_csv(args.output_dir / "predictions.tsv", sep="\t", index=False)
    score = pd.DataFrame(
        [
            {
                "n": len(predictions),
                "pearson_r": np.corrcoef(predictions.age, predictions.prediction)[0, 1],
                "mae": predictions.absolute_error.mean(),
            }
        ]
    )
    score.to_csv(args.output_dir / "score.tsv", sep="\t", index=False, float_format="%.6f")
    print(score.to_string(index=False))
