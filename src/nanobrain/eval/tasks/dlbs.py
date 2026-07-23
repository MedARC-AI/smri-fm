"""DLBS wave-1 T1w eval tasks, streamed from OpenNeuro ds004856 on S3.

A fixed list of wave-1 T1w scan paths (resources/dlbs_wave1_t1w_images.txt) is joined to
the study's participants.tsv for age and sex labels.
"""

import importlib.resources
import io

import fsspec
import pandas as pd
from datasets import Dataset, Features, Nifti, Value

from nanobrain.eval.tasks import register_task
from nanobrain.eval.tasks.base import ClassificationTask, RegressionTask

ROOT = "openneuro.org/ds004856"
SEX_MAP = {"f": 0, "m": 1}


def _t1w_paths() -> list[str]:
    resource = importlib.resources.files("nanobrain.eval.tasks") / "resources"
    return (resource / "dlbs_wave1_t1w_images.txt").read_text().strip().splitlines()


def _generate_dlbs(paths: list[str]):
    fs = fsspec.filesystem("s3", anon=True)
    participants = pd.read_csv(
        io.BytesIO(fs.cat_file(f"{ROOT}/participants.tsv")), sep="\t"
    ).set_index("participant_id")
    for path in paths:
        sub = path.split("/")[0]
        row = participants.loc[sub]
        image = fs.cat_file(f"{ROOT}/{path}")
        yield {
            "participant_id": sub,
            "age": int(row["AgeMRI_W1"]),
            "sex": row["Sex"],
            "image": {"path": None, "bytes": image},
        }


def load_dlbs() -> Dataset:
    features = Features(
        {
            "participant_id": Value("string"),
            "age": Value("int32"),
            "sex": Value("string"),
            "image": Nifti(),
        }
    )
    return Dataset.from_generator(
        _generate_dlbs, features=features, gen_kwargs={"paths": _t1w_paths()}, num_proc=8
    )


@register_task
def dlbs_age() -> RegressionTask:
    return RegressionTask(name="dlbs_age", dataset_fn=load_dlbs, target_col="age")


@register_task
def dlbs_sex() -> ClassificationTask:
    # Sex is 'f'/'m' in participants.tsv; map male -> 1 as the positive class.
    return ClassificationTask(
        name="dlbs_sex", dataset_fn=load_dlbs, target_col="sex", target_map=SEX_MAP
    )
