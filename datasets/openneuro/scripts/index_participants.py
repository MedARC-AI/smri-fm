import json
import yaml
from pathlib import Path

import pandas as pd

SEX_MAP = {"m": "M", "male": "M", "man": "M", "f": "F", "female": "F", "woman": "F"}

# special cases
AGE_COLS = {
    "ds000144": "ScanAge",
    "ds001486": "age_ses-T1",
    "ds001894": "age_ses-T1_T1w",
    "ds002424": "age_ses-T1",
    "ds002886": "age_ses-T1",
    "ds003425": "Age_Sess1",
    "ds003710": "age_ses-1",
    "ds005027": "ScanAge",
}

AGE_UNITS = {
    "ds006169": "months",
}


def normalize_sub(val: str):
    try:
        return val.removeprefix("sub-")
    except AttributeError:
        return pd.NA


def normalize_ses(val: str):
    try:
        return val.removeprefix("ses-")
    except AttributeError:
        return pd.NA


def normalize_sex(val):
    try:
        return SEX_MAP.get(str(val).strip().lower(), pd.NA)
    except (ValueError, TypeError):
        return pd.NA


def normalize_age(val):
    try:
        return round(float(val), 2)
    except (ValueError, TypeError):
        return pd.NA


def get_age_units(ds_root: Path, age_col: str = "age"):
    path = ds_root / "participants.json"
    age_col = age_col.lower()

    dataset = ds_root.name
    if dataset in AGE_UNITS:
        return AGE_UNITS[dataset]

    units = "years"
    try:
        if path.exists():
            meta = json.loads(path.read_text())
            meta = {k.lower(): v for k, v in meta.items()}
            if age_col in meta:
                age_info = json.dumps(meta[age_col]).lower()
                if "month" in age_info:
                    units = "months"
                    print(f"override age {ds_root.name} {units=} {age_info}")
    except Exception as e:
        print(f"age parse error {path}: {e!r}")
    return units


def load_participants(path: Path, dataset: str) -> pd.DataFrame | None:
    df = pd.read_csv(path, sep="\t", dtype=str)
    cols_lower = {c.lower(): c for c in df.columns}

    if "participant_id" not in df:
        return None

    ses_col = cols_lower.get("session") or cols_lower.get("session_id")
    sex_col = cols_lower.get("sex") or cols_lower.get("gender")
    age_col = AGE_COLS.get(dataset) or cols_lower.get("age")

    out = pd.DataFrame()
    out["dataset"] = len(df) * [dataset]
    out["sub"] = df["participant_id"].map(normalize_sub)
    out["ses"] = df[ses_col].map(normalize_ses) if ses_col else pd.NA
    out["sex"] = df[sex_col].map(normalize_sex) if sex_col else pd.NA
    out["age"] = df[age_col].map(normalize_age) if age_col else pd.NA
    out = out.dropna(subset=["sub"])

    # handle age units
    if age_col:
        age_units = get_age_units(path.parent, age_col)
        scale = {"years": 1.0, "months": 12.0}[age_units]
        out["age"] = out["age"] / scale

    # handle invalid ages
    out.loc[out["age"] <= 0, "age"] = pd.NA
    age_max = out["age"].max()
    if age_max > 85:
        print(f"large age max {dataset} {age_max:.0f}")

    # handle multiple ages
    sub_age_range = out.groupby(["sub", "ses"], dropna=False).agg(
        {"age": lambda x: x.max() - x.min()}
    )
    max_age_range = sub_age_range["age"].max()
    if max_age_range > 0:
        print(f"multiple ages {dataset} max range={max_age_range:.1f}y")
    out = out.drop_duplicates(subset=["sub", "ses"])
    return out


def load_participants_ds004856(path: Path, dataset: str):
    # special loader for DLBS bc this is a nice longitudinal dataset
    df = pd.read_csv(path, sep="\t", dtype=str)
    out = df[["participant_id", "Sex", "AgeMRI_W1", "AgeMRI_W2", "AgeMRI_W3"]].copy()
    out.columns = ["sub", "sex", "age-wave1", "age-wave2", "age-wave3"]
    out = pd.wide_to_long(out, ["age"], i="sub", j="ses", sep="-", suffix="wave[1-3]")
    out = out.reset_index()
    out = out[["sub", "ses", "sex", "age"]]
    out.insert(0, "dataset", dataset)
    out["sub"] = out["sub"].map(normalize_sub)
    out["sex"] = out["sex"].map(normalize_sex)
    out["age"] = out["age"].map(normalize_age)
    out = out.dropna(subset="age")
    out = out.sort_values(["sub", "ses"])
    return out


# Override loader for special cases
DS_LOADERS = {
    "ds004856": load_participants_ds004856,
}


def main():
    root = Path("data/openneuro")
    outpath = Path("metadata/openneuro_participants.csv")

    with Path("metadata/openneuro_exclude_datasets.yaml").open() as f:
        exclude = set(yaml.safe_load(f))

    frames = []
    for tsv in sorted(root.glob("*/participants.tsv")):
        dataset = tsv.parent.name
        if dataset in exclude:
            continue

        try:
            loader = DS_LOADERS.get(dataset, load_participants)
            result = loader(tsv, dataset)
            if result is not None:
                frames.append(result)
        except Exception as e:
            print(f"skip {tsv}: {e!r}")

    df = pd.concat(frames, ignore_index=True)
    df.to_csv(outpath, index=False)
    print(f"Saved {len(df)} rows to {outpath}")
    print(df[["sub", "sex", "age"]].count())


if __name__ == "__main__":
    main()
