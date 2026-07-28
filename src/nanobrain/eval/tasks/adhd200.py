"""ADHD-200 ADHD-vs-control eval task, streamed from the fcp-indi mirror on S3.

The three ADHD subtypes (combined, inattentive, hyperactive/impulsive) are pooled into one
positive class against typically developing controls. As with ABIDE the confound that matters
is the acquisition site, so each case is paired with a control from the same (site, age band,
sex) cell, leaving site, age and sex at chance; the cohort is capped at MAX_SUBJECTS out of a
480-subject matched pool, and Pittsburgh drops out entirely for want of pairable cases.

Two of the ten sites are dropped for missing labels rather than for anything about their scans:
Brown is the unlabelled test release (no `dx` column at all) and WashU has controls only. Scans
flagged by the study's own anatomical QC are excluded, which costs ~110 subjects; controls fail
QC slightly more often than cases within a site (93% vs 97% pass), so this is a mild bias, not
the load-bearing shortcut that the equivalent flag was in CNP.
"""

import importlib.resources
import io

import fsspec
import numpy as np
import pandas as pd
from datasets import Dataset, Features, Nifti, Value

from nanobrain.eval.tasks import register_task
from nanobrain.eval.tasks.base import ClassificationTask
from nanobrain.eval.tasks.utils import matched_indices

ROOT = "fcp-indi/data/Projects/ADHD200/RawDataBIDS"
SITES = (
    "KKI", "NYU", "NeuroIMAGE", "OHSU", "Peking_1", "Peking_2", "Peking_3", "Pittsburgh",
)  # fmt: skip
CONTROL_DX = "Typically Developing Children"
AGE_BINS = (12, 15, 18)
MATCH_SEED = 0
MAX_SUBJECTS = 250
DX_MAP = {"control": 0, "adhd": 1}


def _t1w_paths() -> dict[str, str]:
    """`<site>/<sub>` -> path. Session and acq entities vary, so the paths are listed, not built."""
    resource = importlib.resources.files("nanobrain.eval.tasks") / "resources"
    lines = (resource / "adhd200_t1w_images.txt").read_text().strip().splitlines()
    return {"/".join(path.split("/")[:2]): path for path in lines}


def _phenotype() -> pd.DataFrame:
    fs = fsspec.filesystem("s3", anon=True)
    frames = []
    for site in SITES:
        table = pd.read_csv(
            io.BytesIO(fs.cat_file(f"{ROOT}/{site}/participants.tsv")), sep="\t", encoding="latin-1"
        )
        table["site"] = site
        frames.append(table[["participant_id", "site", "dx", "age", "gender", "qc_anatomical_1"]])
    return pd.concat(frames, ignore_index=True)


def _cohort() -> list[dict]:
    paths = _t1w_paths()
    pheno = _phenotype()
    pheno["key"] = pheno["site"] + "/sub-" + pheno["participant_id"].astype(str).str.zfill(7)
    usable = (
        pheno["key"].isin(paths)
        & pheno["dx"].notna()
        & pheno["gender"].isin(("Male", "Female"))  # one NYU subject has neither
        & (pheno["qc_anatomical_1"] == "Pass")
    )
    # NYU, Peking_1 and Pittsburgh list 101 subjects twice, once per release, with identical
    # rows; without this the same scan can land in both the train and the test fold.
    pheno = pheno[usable].drop_duplicates("key").reset_index(drop=True)
    pheno["sex"] = np.where(pheno["gender"] == "Male", "M", "F")

    labels = (pheno["dx"] != CONTROL_DX).astype(int).tolist()
    bands = np.digitize(pheno["age"], AGE_BINS).tolist()
    cells = list(zip(pheno["site"], bands, pheno["sex"]))
    keep = matched_indices(cells, labels, seed=MATCH_SEED, cap=MAX_SUBJECTS)

    rows = []
    for index in keep:
        row = pheno.loc[index]
        rows.append(
            {
                "participant_id": row["key"].split("/")[1],
                "site": row["site"],
                "diagnosis": "control" if row["dx"] == CONTROL_DX else "adhd",
                "age": float(row["age"]),
                "sex": row["sex"],
                "path": paths[row["key"]],
            }
        )
    return rows


def _generate_adhd200(rows: list[dict]):
    fs = fsspec.filesystem("s3", anon=True)
    for row in rows:
        image = fs.cat_file(f"{ROOT}/{row['path']}")
        yield {
            "participant_id": row["participant_id"],
            "site": row["site"],
            "diagnosis": row["diagnosis"],
            "age": row["age"],
            "sex": row["sex"],
            "image": {"path": None, "bytes": image},
        }


def load_adhd200() -> Dataset:
    features = Features(
        {
            "participant_id": Value("string"),
            "site": Value("string"),
            "diagnosis": Value("string"),
            "age": Value("float32"),
            "sex": Value("string"),
            "image": Nifti(),
        }
    )
    return Dataset.from_generator(
        _generate_adhd200,
        features=features,
        gen_kwargs={"rows": _cohort()},
        num_proc=8,
    )


@register_task
def adhd200_adhd_control() -> ClassificationTask:
    """Pooled ADHD vs control, matched: 250 subjects, 125 per class, 7 sites."""
    return ClassificationTask(
        name="adhd200_adhd_control",
        dataset_fn=load_adhd200,
        target_col="diagnosis",
        target_map=DX_MAP,
    )
