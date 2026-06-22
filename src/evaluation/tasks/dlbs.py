import importlib.resources
import io

import fsspec
import pandas as pd
from datasets import Dataset, Features, Nifti, Value

from evaluation.tasks.column import ColumnTask
from evaluation.tasks.registry import register_task

ROOT = "openneuro.org/ds004856"


def load_dlbs_t1w() -> Dataset:
    files = importlib.resources.files("evaluation.tasks.resources")
    with files.joinpath("dlbs_wave1_t1w_images.txt").open() as f:
        paths = f.read().strip().splitlines()

    columns = {
        "AgeMRI_W1": Value("int32"),
        "Sex": Value("string"),
        "EduComp": Value("int32"),
        "BMI_W1": Value("float32"),
        "MMSE_W1": Value("float32"),
    }

    dataset = Dataset.from_generator(
        _generate_dlbs_t1w_samples,
        features=Features(
            {
                "participant_id": Value("string"),
                **columns,
                "path": Value("string"),
                "image": Nifti(),
            }
        ),
        gen_kwargs={"paths": paths, "columns": tuple(columns)},
        num_proc=8,
    )
    return dataset


def _generate_dlbs_t1w_samples(paths: list[str], columns: tuple[str]):
    fs = fsspec.filesystem("s3", anon=True)

    participants = pd.read_csv(
        io.BytesIO(fs.cat_file(f"{ROOT}/participants.tsv")),
        sep="\t",
        dtype_backend="pyarrow",
    )
    participants = participants.set_index("participant_id")

    for path in paths:
        sub = path.split("/")[0]
        row = participants.loc[sub, :].to_dict()
        buf = fs.cat_file(f"{ROOT}/{path}")

        record = {
            "participant_id": sub,
            **{col: row[col] for col in columns},
            "path": path,
            "image": {"path": None, "bytes": buf},
        }
        yield record


@register_task
def dlbs_age(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    return ColumnTask(
        name="dlbs_age",
        kind="regression",
        data=load_dlbs_t1w(),
        n_splits=n_splits,
        seed=seed,
        target_column="AgeMRI_W1",
        group_column="participant_id",
    )


@register_task
def dlbs_sex(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    return ColumnTask(
        name="dlbs_sex",
        kind="classification",
        data=load_dlbs_t1w(),
        n_splits=n_splits,
        seed=seed,
        target_column="Sex",
        group_column="participant_id",
    )
