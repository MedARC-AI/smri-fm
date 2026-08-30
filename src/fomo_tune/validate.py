"""Check a packaged `.sif` against the run it came from.

    uv run python -m fomo_tune.validate <run_dir>

Needs apptainer and a GPU to run.
"""

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

HERE = Path(__file__).parent
VALIDATOR = "third_party/container-validator/container_validator/validate.py"

# inputs and outputs for each task.
TASK_IO = {
    "task1": (
        {
            "flair": "ses-01/flair.nii.gz",
            "adc": "ses-01/adc.nii.gz",
            "dwi": "ses-01/dwi_b1000.nii.gz",
        },
        "prediction.txt",
    ),
    "task2": (
        {"flair": "ses-01/flair.nii.gz", "dwi": "ses-01/dwi_b1000.nii.gz"},
        "prediction.nii.gz",
    ),
    "task3": ({"t1": "ses-01/t1w.nii.gz"}, "prediction.txt"),
    "task4": ({"t2": "ses-01/t2w.nii.gz"}, "prediction.nii.gz"),
    "task5": ({"t1": "ses_01/t1.nii.gz"}, "prediction.txt"),
    "task6_and_7": ({"input": "ses-01/flair.nii.gz"}, "prediction.npy"),
}

# the eval folder a task's subjects come from, where it is not Task_<k>. Tasks 6 and 7 embed one
# image of any modality, so they borrow task 1's.
TASK_DATA = {"task6_and_7": "Task_1"}


def versions(python: list[str], packages: list[str]) -> dict[str, str]:
    script = (
        "from importlib.metadata import version, PackageNotFoundError\n"
        f"for name in {packages!r}:\n"
        "    try: print(name, version(name))\n"
        "    except PackageNotFoundError: print(name, 'MISSING')"
    )
    out = subprocess.run([*python, "-c", script], capture_output=True, text=True, check=True)
    return dict(line.split() for line in out.stdout.split("\n") if line)


def check_pins(sif: Path) -> bool:
    """Every pin in the def, against what the container and this environment actually import."""
    pinned = dict(
        re.findall(r"^\s+([\w.-]+)==([\w.]+)", (HERE / "Apptainer.def").read_text(), re.M)
    )
    container = versions(["apptainer", "exec", str(sif), "python"], list(pinned))
    host = versions([sys.executable], list(pinned))

    print(f"\n{'package':<18} {'def':>10} {'container':>12} {'host':>10}")
    ok = True
    for name, pin in pinned.items():
        agree = pin == container[name] == host[name]
        ok &= agree
        print(
            f"{name:<18} {pin:>10} {container[name]:>12} {host[name]:>10}"
            f"{'' if agree else '   <- MISMATCH'}"
        )
    return bool(ok)


def check_challenge_validator(task: str, sif: Path) -> bool:
    out = subprocess.run(
        [sys.executable, VALIDATOR, "--task", task, "--sif", str(sif)],
        capture_output=True,
        text=True,
    )
    print("\n" + "\n".join(out.stdout.strip().split("\n")[-3:]))
    return out.returncode == 0


def read_output(path: Path) -> np.ndarray:
    if path.suffixes[-2:] == [".nii", ".gz"]:
        import nibabel as nib

        return np.asarray(nib.load(path).dataobj)
    if path.suffix == ".npy":
        return np.load(path)
    return np.array([float(path.read_text().strip())])


def check_host_against_container(
    run_dir: Path, sif: Path, task: str, data_root: Path, n_subjects: int
) -> bool:
    """The same subjects through the real `predict` entrypoint on both sides.

    Both read the same copied files, bound at `/input` as the challenge hands them over, so the
    only thing that differs is which environment runs them.
    """
    inputs, output_name = TASK_IO[task]
    folder = TASK_DATA.get(task, f"Task_{task.removeprefix('task')}")
    task_dir = data_root / folder / "preprocessed"
    subjects = sorted(p for p in task_dir.iterdir() if p.is_dir())[:n_subjects]

    print(f"\n{'subject':<12} {'host':>14} {'container':>14} {'max |diff|':>12}")
    ok = True
    for subject in subjects:
        with tempfile.TemporaryDirectory(dir=run_dir) as tmp:
            stage = Path(tmp)
            (stage / "input").mkdir()
            (stage / "output").mkdir()
            for flag, relative in inputs.items():
                shutil.copy(subject / relative, stage / "input" / f"{flag}.nii.gz")

            host_flags = []
            container_flags = []
            for flag in inputs:
                host_flags += [f"--{flag}", str(stage / "input" / f"{flag}.nii.gz")]
                container_flags += [f"--{flag}", f"/input/{flag}.nii.gz"]

            host_output = stage / "output" / f"host_{output_name}"
            container_output = stage / "output" / output_name

            subprocess.run(
                [sys.executable, "-m", f"fomo_tune.main_{task}", "predict", *host_flags]
                + ["--output", str(host_output), "--model-dir", str(run_dir / "model")],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["apptainer", "run", "--nv"]
                + ["--bind", f"{stage / 'input'}:/input", "--bind", f"{stage / 'output'}:/output"]
                + [str(sif), *container_flags, "--output", f"/output/{output_name}"],
                check=True,
                capture_output=True,
            )

            host = read_output(host_output)
            container = read_output(container_output)
            difference = np.abs(host - container).max()
            ok &= bool(difference == 0)
            print(
                f"{subject.name:<12} {host.ravel()[0]:>14.6f} {container.ravel()[0]:>14.6f} "
                f"{difference:>12.3e}"
            )
    return bool(ok)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path, help="a train run dir, holding model/ and the sif")
    parser.add_argument("--sif", type=Path, help="defaults to <run_dir>/<task>.sif")
    parser.add_argument("--data-root", type=Path, default=Path("data/fomo_eval"))
    parser.add_argument("--n-subjects", type=int, default=3)
    args = parser.parse_args()

    task = OmegaConf.load(args.run_dir / "model/config.yaml").task
    sif = args.sif or args.run_dir / f"{task}.sif"

    results = {
        "pins": check_pins(sif),
        "challenge validator": check_challenge_validator(task, sif),
        "host vs container": check_host_against_container(
            args.run_dir, sif, task, args.data_root, args.n_subjects
        ),
    }

    print()
    for name, passed in results.items():
        print(f"{'PASS' if passed else 'FAIL'}  {name}")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
