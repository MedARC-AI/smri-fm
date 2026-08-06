"""UCLA CNP (LA5c) T1w eval tasks, streamed from OpenNeuro ds000030 on S3.

Diagnosis, age and sex come from participants.tsv, along with two confounds that track diagnosis
hard: imaging site (controls sit 4:1 across the two scanners) and the headset ghost artifact in
~20% of scans (40% of schizophrenia vs 12% of controls). With age, ~5 years older in the patient
groups, they are worth up to 0.69 AUROC alone, so each classification cohort is subsampled to equal
class size within every (scanner, ghost, age decade) cell, putting all three at chance. Sizes are
deterministic; MATCH_SEED only picks which surplus subjects are kept.
"""

import io

import fsspec
import numpy as np
import pandas as pd
from datasets import Dataset, Features, Nifti, Value

from nanobrain.eval.tasks import register_task
from nanobrain.eval.tasks.base import ClassificationTask
from nanobrain.eval.tasks.utils import matched_indices

ROOT = "openneuro.org/ds000030"
SEX_MAP = {"F": 0, "M": 1}
MATCH_SEED = 0
AGE_BINS = (21, 31, 41, 51)


def load_participants() -> pd.DataFrame:
    fs = fsspec.filesystem("s3", anon=True)
    table = pd.read_csv(io.BytesIO(fs.cat_file(f"{ROOT}/participants.tsv")), sep="\t")
    return table[table["T1w"] == 1].set_index("participant_id")


def generate_cnp(subject_ids: list[str]):
    fs = fsspec.filesystem("s3", anon=True)
    participants = load_participants()
    for sub in subject_ids:
        row = participants.loc[sub]
        image = fs.cat_file(f"{ROOT}/{sub}/anat/{sub}_T1w.nii.gz")
        yield {
            "participant_id": sub,
            "diagnosis": row["diagnosis"],
            "age": int(row["age"]),
            "sex": row["gender"],
            "scanner": str(int(row["ScannerSerialNumber"])),
            "ghost": row["ghost_NoGhost"] == "ghost",
            "image": {"path": None, "bytes": image},
        }


def load_cnp() -> Dataset:
    features = Features(
        {
            "participant_id": Value("string"),
            "diagnosis": Value("string"),
            "age": Value("int32"),
            "sex": Value("string"),
            "scanner": Value("string"),
            "ghost": Value("bool"),
            "image": Nifti(),
        }
    )
    subject_ids = sorted(load_participants().index)
    return Dataset.from_generator(
        generate_cnp,
        features=features,
        gen_kwargs={"subject_ids": subject_ids},
        num_proc=8,
    )


def load_cohort(diagnoses: tuple[str, ...]) -> Dataset:
    """The subjects in the given diagnosis groups. Reading a column does not decode images."""
    dataset = load_cnp()
    keep = [i for i, dx in enumerate(dataset["diagnosis"]) if dx in diagnoses]
    return dataset.select(keep)


def matched_cohort(diagnoses: tuple[str, ...], target_col: str, target_map: dict) -> Dataset:
    cohort = load_cohort(diagnoses)
    labels = [target_map[value] for value in cohort[target_col]]
    decades = np.digitize(cohort["age"], AGE_BINS).tolist()
    cells = list(zip(cohort["scanner"], cohort["ghost"], decades))
    return cohort.select(matched_indices(cells, labels, seed=MATCH_SEED))


@register_task
def cnp_sex() -> ClassificationTask:
    """Sex over controls, matched: 102 subjects, 51 per class."""
    return ClassificationTask(
        name="cnp_sex",
        dataset_fn=lambda: matched_cohort(("CONTROL",), "sex", SEX_MAP),
        target_col="sex",
        target_map=SEX_MAP,
    )


@register_task
def cnp_adhd_control() -> ClassificationTask:
    """ADHD vs control, matched: 76 subjects, 38 per class -- only large effects will show."""
    target_map = {"CONTROL": 0, "ADHD": 1}
    return ClassificationTask(
        name="cnp_adhd_control",
        dataset_fn=lambda: matched_cohort(("CONTROL", "ADHD"), "diagnosis", target_map),
        target_col="diagnosis",
        target_map=target_map,
    )


@register_task
def cnp_schz_bipolar_control() -> ClassificationTask:
    """Schizophrenia and bipolar pooled against control, as in Neuro-JEPA. Matched: 126, 63/63."""
    target_map = {"CONTROL": 0, "SCHZ": 1, "BIPOLAR": 1}
    return ClassificationTask(
        name="cnp_schz_bipolar_control",
        dataset_fn=lambda: matched_cohort(("CONTROL", "SCHZ", "BIPOLAR"), "diagnosis", target_map),
        target_col="diagnosis",
        target_map=target_map,
    )
