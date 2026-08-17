#!/usr/bin/env bash
# Run on a Narval LOGIN node. Compute nodes have no internet, so the checkpoint,
# the task 3 zip and every wheel have to be on disk before the job starts.
#
#   export SLURM_ACCOUNT=def-<supervisor>
#   bash experiments/task7_pooling_bench/prefetch_narval.sh
#
# Then: sbatch experiments/task7_pooling_bench/launch_narval.sh
#
# Login nodes have no GPU, so nothing here touches CUDA. Takes a while: the
# task 3 zip is 494 preprocessed T1w volumes.

set -euo pipefail

: "${PROJECT:?PROJECT is not set; are you on Narval?}"
: "${SCRATCH:?SCRATCH is not set; are you on Narval?}"

# Code, venv and weights live in PROJECT (large, not purged). Job I/O goes to
# SCRATCH, which is purged after ~60 days idle -- a venv there disappears
# between rounds of work.
VENV="${PROJECT}/fomo_task7_venv"
CACHE="${SCRATCH}/fomo_task7"
mkdir -p "${CACHE}/eval"

module load StdEnv/2023 gcc/12.3 cuda/12.2 cudnn/8.9 arrow/21.0.0 python/3.11

# HF_HOME is not exported in a fresh login shell even after a successful
# huggingface-cli login, and a different login node starts without it, so the
# CLI reports "not logged in" while a valid token sits on disk.
export HF_HOME="${HF_HOME:-${PROJECT}/hf_cache}"
export HF_HUB_DISABLE_XET=1        # Xet transfers fail on this cluster
mkdir -p "${HF_HOME}"

if [[ ! -d "${VENV}" ]]; then
    virtualenv --no-download "${VENV}"
fi
source "${VENV}/bin/activate"
pip install --no-index --upgrade pip

# From the Compute Canada wheelhouse where possible.
pip install --no-index torch numpy scipy scikit-learn nibabel

# Not in the wheelhouse; the login node still has a route to PyPI. The bench
# does not need asparagus, matplotlib or the rest of pyproject -- only the
# import set of cache_pooled/bench_task7 and what backbone.py pulls in.
pip install huggingface_hub datasets einops jaxtyping timm omegaconf fsspec

python - <<PY
import os, pathlib
from huggingface_hub import hf_hub_download
ckpt = hf_hub_download(
    "medarc/walnut",
    "checkpoints/walnut-v0-1/vitl/sub-52k/checkpoint-last.pth",
)
print("checkpoint cached:", ckpt)
pathlib.Path("${CACHE}/ckpt_path.txt").write_text(ckpt)
PY

# open_zip() opens a path that exists rather than downloading, so pointing
# FOMO_EVAL_BASE_URL at this directory keeps the loader on local disk.
BASE="${FOMO_EVAL_BASE_URL:-https://sid.erda.dk/share_redirect/fmeuvo1EdF}"
if [[ ! -f "${CACHE}/eval/Task_3.zip" ]]; then
    echo "downloading Task_3.zip (494 volumes, this is the slow part)"
    curl -L --fail --retry 3 -o "${CACHE}/eval/Task_3.zip.part" "${BASE}/Task_3.zip"
    mv "${CACHE}/eval/Task_3.zip.part" "${CACHE}/eval/Task_3.zip"
fi

python - <<PY
import sys, zipfile
z = zipfile.ZipFile("${CACHE}/eval/Task_3.zip")
subs = sorted({n.split("/")[2] for n in z.namelist() if n.endswith(".nii.gz")})
print(f"Task_3.zip: {len(subs)} subjects")
sys.exit(0 if len(subs) == 494 else 1)
PY

cat <<EOF

prefetch complete
  venv        ${VENV}
  checkpoint  $(cat "${CACHE}/ckpt_path.txt")
  eval data   ${CACHE}/eval
  HF_HOME     ${HF_HOME}

next:
  export SLURM_ACCOUNT=def-<supervisor>
  sbatch experiments/task7_pooling_bench/launch_narval.sh
EOF
