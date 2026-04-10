import subprocess
from pathlib import Path

import nibabel as nib
import pandas as pd


def main():
    root = Path("data/openneuro")
    listpath = Path("metadata/openneuro_filelist.txt")
    if listpath.exists():
        print(f"loading file list: {listpath}")
        path_list = listpath.read_text().strip().splitlines()
        path_list = [Path(p) for p in path_list]
    else:
        print("scanning for nifti images...")
        path_list = sorted([p.relative_to(root) for p in root.rglob("*.nii*")])
        path_list = [
            p for p in path_list if not p.name.startswith(".") and "bidsignore" not in str(p)
        ]
        with listpath.open("w") as f:
            print("\n".join(map(str, path_list)), file=f)
    print(f"total images: {len(path_list)}")

    outpath = Path("metadata/openneuro_images.csv")
    print(f"saving to {outpath}")

    if outpath.exists():
        df = pd.read_csv(outpath)
        records = df.to_dict("records")
        completed = set(df["path"])
    else:
        records = []
        completed = set()

    for ii, path in enumerate(path_list):
        if str(path) in completed:
            continue

        try:
            meta = parse_metadata(path)
            info = read_header(root / path)
        except Exception as e:
            print(f"SKIP {path}: {e}")
            continue

        md5 = md5sum(root / path)
        records.append({**meta, **info, "md5": md5, "path": str(path)})

        if (ii + 1) % 1000 == 0 or (ii + 1) == len(path_list):
            print(f"saving {ii + 1}/{len(path_list)}, total: {len(records)}", flush=True)
            save_records(records, outpath)


def parse_metadata(path: Path):
    # ds000001/sub-04/anat/sub-04_T1w.nii.gz
    dataset = path.parts[0]
    datatype = path.parts[-2]
    assert datatype == "anat", f"invalid {datatype=}"
    stem, ext = path.name.split(".", 1)
    stem, suffix = stem.rsplit("_", 1)
    meta = dict(item.split("-") for item in stem.split("_") if "-" in item)
    meta = {"dataset": dataset, **meta, "suffix": suffix}
    return meta


def read_header(path: Path):
    img = nib.load(path, mmap=True)
    shape = list(img.header.get_data_shape())
    pixdim = [round(float(v), 2) for v in img.header.get_zooms()]
    dtype = img.get_data_dtype().name
    orient = "".join(nib.aff2axcodes(img.affine))
    size = path.stat().st_size
    info = {"shape": shape, "pixdim": pixdim, "dtype": dtype, "orient": orient, "size": size}
    return info


def md5sum(path: Path):
    try:
        s = subprocess.check_output(["md5sum", str(path)]).decode()
        s = s.strip().split()[0]
    except Exception:
        s = None
    return s


def save_records(records: list[dict], outpath: Path):
    df = pd.DataFrame.from_records(records)
    cols = ["suffix", "shape", "pixdim", "dtype", "orient", "size", "md5", "path"]
    for col in cols:
        df[col] = df.pop(col)
    df.to_csv(outpath, index=False)


if __name__ == "__main__":
    main()
