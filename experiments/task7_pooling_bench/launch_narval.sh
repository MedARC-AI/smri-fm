#!/usr/bin/env bash
#SBATCH --job-name=fomo_task7_bench
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --output=slurms/slurm-%j.out

# One GPU pass over task 3's 494 subjects applying every pooling, then the
# (pooling x transform) grid on the resulting cache.
#
#   bash experiments/task7_pooling_bench/prefetch_narval.sh   # login node
#   sbatch --account=def-<supervisor> \
#          experiments/task7_pooling_bench/launch_narval.sh
#   squeue -u $USER   /   seff <jobid>
#
# Pass --account on the command line (or export SBATCH_ACCOUNT) rather than
# hardcoding it, so a supervisor's name is not committed to a repo that may go
# upstream. Note SBATCH_ACCOUNT, not SLURM_ACCOUNT: the latter is an output
# variable Slurm sets inside the job and is ignored at submission.
# --mem is system RAM, not VRAM.

set -euo pipefail

: "${PROJECT:?PROJECT is not set}"
: "${SCRATCH:?SCRATCH is not set}"

VENV="${PROJECT}/fomo_task7_venv"
CACHE="${SCRATCH}/fomo_task7"
EXP_DIR="experiments/task7_pooling_bench"
mkdir -p slurms "${EXP_DIR}/output"

if [[ ! -d "${VENV}" || ! -f "${CACHE}/ckpt_path.txt" ]]; then
    echo "prefetch has not run: expected ${VENV} and ${CACHE}/ckpt_path.txt" >&2
    echo "run prefetch_narval.sh on a login node first" >&2
    exit 1
fi

module load StdEnv/2023 gcc/12.3 cuda/12.2 cudnn/8.9 arrow/21.0.0 python/3.11
source "${VENV}/bin/activate"

# Offline. Without these a missing prefetch stalls on a socket the compute node
# cannot open, and the job burns its whole walltime doing nothing instead of
# failing in the first minute.
export HF_HOME="${HF_HOME:-${PROJECT}/hf_cache}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export HF_DATASETS_CACHE="${SLURM_TMPDIR:-${CACHE}}/hf_datasets"
export FOMO_EVAL_BASE_URL="${CACHE}/eval"
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

CKPT="$(cat "${CACHE}/ckpt_path.txt")"
POOLED="${EXP_DIR}/output/pooled_walnut_v0_1.npz"

echo "=== environment ==="
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=== self-test (no model, no data) ==="
python -m fomo_tune.bench_task7 --sandbox

if [[ ! -f "${POOLED}" ]]; then
    # Smoke 8 subjects first. A bad checkpoint path or a broken transform then
    # fails in about a minute rather than 40, which matters when the queue wait
    # already cost hours.
    echo "=== smoke: 8 subjects ==="
    python -m fomo_tune.cache_pooled \
        --out "${EXP_DIR}/output/smoke.npz" --ckpt-path "${CKPT}" --limit 8
    python -m fomo_tune.bench_task7 --cache "${EXP_DIR}/output/smoke.npz" >/dev/null
    echo "smoke passed"

    echo "=== caching pooled embeddings, all 494 (the only GPU step) ==="
    python -m fomo_tune.cache_pooled --out "${POOLED}" --ckpt-path "${CKPT}"
else
    echo "pooled cache exists; skipping the GPU step"
fi

echo "=== grid ==="
python -m fomo_tune.bench_task7 \
    --cache "${POOLED}" \
    --out "${EXP_DIR}/output/grid.json" \
    | tee "${EXP_DIR}/output/grid.txt"

cat <<EOF

done
  cache  ${POOLED}
  table  ${EXP_DIR}/output/grid.txt
  json   ${EXP_DIR}/output/grid.json

baseline to beat (mean + identity, walnut-v0.1 vitl/sub-52k):
  MAE 3.50   r 0.968   age-bin MAE spread 3.30
EOF
