"""ABIDE I autism-vs-control eval task, streamed from the fcp-indi mirror on S3.

Twenty-odd acquisition sites differ in scanner, protocol and cohort makeup, so site is the
confound that matters here; age (6-64, median 15) and the 85% male skew are the others. Each
autistic subject is therefore paired with a control from the same (site, age band, sex) cell,
which leaves all three at chance, and the cohort is capped at MAX_SUBJECTS to keep the eval
cheap -- the full matched pool is 846, so the cap costs balance nothing.
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

ROOT = "fcp-indi/data/Projects/ABIDE/RawDataBIDS"
SITES = (
    "CMU_a", "CMU_b", "Caltech", "KKI", "Leuven_1", "Leuven_2", "MaxMun_a", "MaxMun_b",
    "MaxMun_c", "MaxMun_d", "NYU", "OHSU", "Olin", "Pitt", "SBL", "SDSU", "Stanford",
    "Trinity", "UCLA_1", "UCLA_2", "UM_1", "UM_2", "USM", "Yale",
)  # fmt: skip
AGE_BINS = (12, 15, 18)
MATCH_SEED = 0
MAX_SUBJECTS = 250
DX_MAP = {"control": 0, "autism": 1}


def t1w_paths() -> dict[str, str]:
    """`<site>/<sub>` -> path, which also pins which subjects actually have a scan."""
    resource = importlib.resources.files("nanobrain.eval.tasks") / "resources"
    lines = (resource / "abide_t1w_images.txt").read_text().strip().splitlines()
    return {"/".join(path.split("/")[:2]): path for path in lines}


def phenotype() -> pd.DataFrame:
    fs = fsspec.filesystem("s3", anon=True)
    frames = []
    for site in SITES:
        table = pd.read_csv(
            io.BytesIO(fs.cat_file(f"{ROOT}/{site}/participants.tsv")), sep="\t", encoding="latin-1"
        )
        table["site"] = site
        frames.append(table[["participant_id", "site", "DX_GROUP", "AGE_AT_SCAN", "SEX"]])
    return pd.concat(frames, ignore_index=True)


def load_cohort() -> list[dict]:
    paths = t1w_paths()
    pheno = phenotype()
    pheno["key"] = pheno["site"] + "/sub-" + pheno["participant_id"].astype(str).str.zfill(7)
    pheno = pheno[pheno["key"].isin(paths)].reset_index(drop=True)

    # DX_GROUP is 1 for autism, 2 for control; SEX is 1 for male.
    pheno["sex"] = np.where(pheno["SEX"] == 1, "M", "F")

    labels = (pheno["DX_GROUP"] == 1).astype(int).tolist()
    bands = np.digitize(pheno["AGE_AT_SCAN"], AGE_BINS).tolist()
    cells = list(zip(pheno["site"], bands, pheno["sex"]))
    keep = matched_indices(cells, labels, seed=MATCH_SEED, cap=MAX_SUBJECTS)

    rows = []
    for index in keep:
        row = pheno.loc[index]
        rows.append(
            {
                "participant_id": row["key"].split("/")[1],
                "site": row["site"],
                "diagnosis": "autism" if row["DX_GROUP"] == 1 else "control",
                "age": float(row["AGE_AT_SCAN"]),
                "sex": row["sex"],
                "path": paths[row["key"]],
            }
        )
    return rows


def generate_abide(rows: list[dict]):
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


def load_abide() -> Dataset:
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
        generate_abide,
        features=features,
        gen_kwargs={"rows": load_cohort()},
        num_proc=8,
    )


@register_task
def abide_autism_control() -> ClassificationTask:
    """Autism vs control, matched: 250 subjects, 125 per class, 23 sites."""
    return ClassificationTask(
        name="abide_autism_control",
        dataset_fn=load_abide,
        target_col="diagnosis",
        target_map=DX_MAP,
    )
