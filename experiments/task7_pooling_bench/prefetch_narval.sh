#!/usr/bin/env bash
# Run this on a Narval LOGIN node. Compute nodes have no internet, so everything
# the job touches has to be on disk first: the checkpoint, the task 3 zip, and
# every wheel.
#
#   bash experiments/task7_pooling_bench/prefetch_narval.sh
#
# Then submit with launch_narval.sh, which runs fully offline.

set -euo pipefail

: "${SCRATCH:?SCRATCH is not set; are you on Narval?}"
CACHE="${SCRATCH}/fomo_task7"
mkdir -p "${CACHE}/eval" "${CACHE}/hf" "${CACHE}/wheels"

export HF_HOME="${CACHE}/hf"

module purge
module load StdEnv/2023 python/3.11

# scipy-stack does not carry scikit-learn on StdEnv/2023, and the compute node
# cannot reach PyPI, so the venv is built here with everything baked in.
if [[ ! -d "${CACHE}/venv" ]]; then
    virtualenv --no-download "${CACHE}/venv"
fi
source "${CACHE}/venv/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index scikit-learn numpy scipy
# These are not in the Compute Canada wheelhouse, so they come from PyPI while
# the login node still has a route to it.
pip install torch huggingface_hub datasets nibabel omegaconf einops fsspec

python - <<'PY'
import os
from huggingface_hub import hf_hub_download
ckpt = hf_hub_download(
    "medarc/walnut",
    "checkpoints/walnut-v0-1/vitl/sub-52k/checkpoint-last.pth",
)
print(f"checkpoint cached: {ckpt}")
open(os.path.join(os.environ["SCRATCH"], "fomo_task7", "ckpt_path.txt"), "w").write(ckpt)
PY

# The eval zip. `open_zip` takes a local path unchanged, so pointing
# FOMO_EVAL_BASE_URL at this directory makes the loader read from disk.
BASE="${FOMO_EVAL_BASE_URL:-https://sid.erda.dk/share_redirect/fmeuvo1EdF}"
if [[ ! -f "${CACHE}/eval/Task_3.zip" ]]; then
    echo "downloading Task_3.zip (this is the big one)"
    curl -L --fail -o "${CACHE}/eval/Task_3.zip" "${BASE}/Task_3.zip"
fi

python -c "
import zipfile, sys
z = zipfile.ZipFile('${CACHE}/eval/Task_3.zip')
subs = sorted({n.split('/')[2] for n in z.namelist() if n.endswith('.nii.gz')})
print(f'Task_3.zip OK: {len(subs)} subjects')
sys.exit(0 if len(subs) == 494 else 1)
" || { echo "Task_3.zip did not unpack to 494 subjects; re-download"; exit 1; }

echo
echo "prefetch complete -> ${CACHE}"
echo "  venv       ${CACHE}/venv"
echo "  checkpoint \$(cat ${CACHE}/ckpt_path.txt)"
echo "  eval data  ${CACHE}/eval"
echo "now: sbatch experiments/task7_pooling_bench/launch_narval.sh"
