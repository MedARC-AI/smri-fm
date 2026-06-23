import shutil
import tempfile
import zipfile
from pathlib import Path

import fsspec
from datasets import Dataset, Features, Nifti, Value

from evaluation.tasks.column import ColumnTask
from evaluation.tasks.metrics import age_bias_correct_predictions, mae, pearson_r, r2
from evaluation.tasks.registry import register_task

# FOMO26 Task 3: brain age prediction from a single T1w volume per subject.
URL = "https://sid.erda.dk/share_redirect/fmeuvo1EdF/Task_3.zip"


def load_fomo_task3() -> Dataset:
    return Dataset.from_generator(
        _generate_fomo_task3_samples,
        features=Features(
            {
                "participant_id": Value("string"),
                "age": Value("int32"),
                "path": Value("string"),
                "image": Nifti(),
            }
        ),
    )


def _generate_fomo_task3_samples():
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "Task_3.zip"
        with fsspec.open(URL) as src, zip_path.open("wb") as dst:
            shutil.copyfileobj(src, dst)

        with zipfile.ZipFile(zip_path) as zf:
            subjects = sorted(
                {name.split("/")[2] for name in zf.namelist() if name.endswith("t1w.nii.gz")}
            )
            for sub in subjects:
                age = int(zf.read(f"Task_3/labels/{sub}/ses-01/labels.txt").strip())
                path = f"preprocessed/{sub}/ses-01/t1w.nii.gz"
                image = zf.read(f"Task_3/{path}")
                yield {
                    "participant_id": sub,
                    "age": age,
                    "path": path,
                    "image": {"path": None, "bytes": image},
                }


@register_task
def fomo_task3_age(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    return ColumnTask(
        name="fomo_task3_age",
        kind="regression",
        data=load_fomo_task3(),
        metrics=(pearson_r, r2, mae),
        n_splits=n_splits,
        seed=seed,
        target_column="age",
        group_column="participant_id",
        prediction_postprocessor=age_bias_correct_predictions,
    )
