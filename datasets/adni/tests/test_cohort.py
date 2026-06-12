import numpy as np
import pandas as pd

from adni_curate import curate

CohortConfig = curate.CohortConfig
build_cohort = curate.build_cohort


def make_manifest():
    rows=[]
    for subject in range(160):
        diagnosis = "CN" if subject % 2 == 0 else "Dementia"
        sex = "Female" if (subject // 2) % 2 == 0 else "Male"
        for session in range(3):
            age = 60 + ((subject + session) % 30)
            rows.append({
                "sample_id": f"sub-{subject:03d}_ses-{session}",
                "participant_id": f"sub-{subject:03d}", "session_id": str(session),
                "scan_date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=365 * session),
                "age": float(age), "sex": sex, "diagnosis": diagnosis,
                "synthseg_volumes": np.arange(101, dtype=float).tolist(),
                "synthseg_qc_mean": .9, "synthseg_qc_min": .8,
                "path": f"images/{subject}-{session}.nii.gz",
            })
    return pd.DataFrame(rows)


def test_build_cohort_is_deterministic_and_subject_exclusive():
    config = CohortConfig(train_size=120, validation_size=40, test_size=40,
                          split_trials=20, seed=7)
    first = build_cohort(make_manifest(), config)
    second = build_cohort(make_manifest(), config)
    pd.testing.assert_frame_equal(first, second)
    assert first["split"].value_counts().to_dict() == {"train": 120, "validation": 40, "test": 40}
    assert first.groupby("participant_id")["split"].nunique().max() == 1
    assert first[first.split == "train"].train_rank.sort_values().tolist() == list(range(120))
    for split, size in [("train", 120), ("validation", 40), ("test", 40)]:
        assert first[first.split == split].diagnosis.value_counts().to_dict() == {"AD": size // 2, "CN": size // 2}
    ranked = first[first.split == "train"].sort_values("train_rank")
    for size in [40, 80, 120]:
        assert ranked.iloc[:size].diagnosis.value_counts().to_dict() == {
            "AD": size // 2, "CN": size // 2,
        }
