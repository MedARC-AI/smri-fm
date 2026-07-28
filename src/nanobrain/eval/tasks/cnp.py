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

ROOT = "openneuro.org/ds000030"
SEX_MAP = {"F": 0, "M": 1}
MATCH_SEED = 0
AGE_BINS = (21, 31, 41, 51)


def _participants() -> pd.DataFrame:
    fs = fsspec.filesystem("s3", anon=True)
    table = pd.read_csv(io.BytesIO(fs.cat_file(f"{ROOT}/participants.tsv")), sep="\t")
    return table[table["T1w"] == 1].set_index("participant_id")


def _generate_cnp(subject_ids: list[str]):
    fs = fsspec.filesystem("s3", anon=True)
    participants = _participants()
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
    subject_ids = sorted(_participants().index)
    return Dataset.from_generator(
        _generate_cnp,
        features=features,
        gen_kwargs={"subject_ids": subject_ids},
        num_proc=8,
    )


def _cohort(diagnoses: tuple[str, ...]) -> Dataset:
    """The subjects in the given diagnosis groups. Reading a column does not decode images."""
    dataset = load_cnp()
    keep = [i for i, dx in enumerate(dataset["diagnosis"]) if dx in diagnoses]
    return dataset.select(keep)


def _match_on_confounds(cohort: Dataset, labels: list[int]) -> Dataset:
    """Equalize the (scanner, ghost, age decade) makeup of the two classes, cell by cell."""
    rng = np.random.default_rng(MATCH_SEED)
    decades = np.digitize(cohort["age"], AGE_BINS)
    cells = list(zip(cohort["scanner"], cohort["ghost"], decades.tolist()))
    keep: list[int] = []
    for cell in sorted(set(cells)):
        in_cell = {
            label: [i for i, (c, y) in enumerate(zip(cells, labels)) if c == cell and y == label]
            for label in (0, 1)
        }
        size = min(len(in_cell[0]), len(in_cell[1]))
        for indices in in_cell.values():
            keep.extend(rng.choice(indices, size=size, replace=False).tolist())
    return cohort.select(sorted(keep))


def _matched_cohort(diagnoses: tuple[str, ...], target_col: str, target_map: dict) -> Dataset:
    cohort = _cohort(diagnoses)
    labels = [target_map[value] for value in cohort[target_col]]
    return _match_on_confounds(cohort, labels)


@register_task
def cnp_sex() -> ClassificationTask:
    """Sex over controls, matched: 102 subjects, 51 per class."""
    return ClassificationTask(
        name="cnp_sex",
        dataset_fn=lambda: _matched_cohort(("CONTROL",), "sex", SEX_MAP),
        target_col="sex",
        target_map=SEX_MAP,
    )


@register_task
def cnp_adhd_control() -> ClassificationTask:
    """ADHD vs control, matched: 76 subjects, 38 per class -- only large effects will show."""
    target_map = {"CONTROL": 0, "ADHD": 1}
    return ClassificationTask(
        name="cnp_adhd_control",
        dataset_fn=lambda: _matched_cohort(("CONTROL", "ADHD"), "diagnosis", target_map),
        target_col="diagnosis",
        target_map=target_map,
    )


@register_task
def cnp_schz_bipolar_control() -> ClassificationTask:
    """Schizophrenia and bipolar pooled against control, as in Neuro-JEPA. Matched: 126, 63/63."""
    target_map = {"CONTROL": 0, "SCHZ": 1, "BIPOLAR": 1}
    return ClassificationTask(
        name="cnp_schz_bipolar_control",
        dataset_fn=lambda: _matched_cohort(("CONTROL", "SCHZ", "BIPOLAR"), "diagnosis", target_map),
        target_col="diagnosis",
        target_map=target_map,
    )
