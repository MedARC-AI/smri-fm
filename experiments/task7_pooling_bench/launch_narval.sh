#!/usr/bin/env bash
#SBATCH --job-name=fomo_task7_bench
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=slurms/slurm-%j.out
#SBATCH --account=def-CHANGEME

# One GPU pass over task 3's 494 subjects, applying every pooling inside it,
# then the (pooling x transform) grid on the resulting cache.
#
# Run prefetch_narval.sh on a LOGIN node first. This job is fully offline: the
# HF_*_OFFLINE flags below make a missed prefetch fail loudly here rather than
# hang on a network call the compute node cannot make.
#
#   sbatch experiments/task7_pooling_bench/launch_narval.sh
#   squeue -u $USER    /    seff <jobid>

set -euo pipefail

: "${SCRATCH:?SCRATCH is not set}"
CACHE="${SCRATCH}/fomo_task7"
EXP_DIR="experiments/task7_pooling_bench"
mkdir -p slurms "${EXP_DIR}/output"

module purge
module load StdEnv/2023 python/3.11 cuda/12.2
source "${CACHE}/venv/bin/activate"

# Offline. Without these a missing prefetch stalls on a socket instead of
# raising, and the job burns its whole walltime doing nothing.
export HF_HOME="${CACHE}/hf"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE="${SLURM_TMPDIR:-${CACHE}}/hf_datasets"

# `open_zip` opens a path that exists rather than fetching, so pointing the base
# at the prefetched directory keeps the loader entirely on local disk.
export FOMO_EVAL_BASE_URL="${CACHE}/eval"
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

CKPT="$(cat "${CACHE}/ckpt_path.txt")"
POOLED="${EXP_DIR}/output/pooled_walnut_v0_1.npz"

echo "=== self-test (no model, no data) ==="
python -m fomo_tune.bench_task7 --sandbox

if [[ ! -f "${POOLED}" ]]; then
    echo "=== caching pooled embeddings (the only GPU step) ==="
    # Smoke 8 subjects first: a broken transform or checkpoint fails in a minute
    # instead of 40, which matters when the queue wait is the expensive part.
    python -m fomo_tune.cache_pooled \
        --out "${EXP_DIR}/output/smoke.npz" --ckpt-path "${CKPT}" --limit 8
    python -m fomo_tune.bench_task7 --cache "${EXP_DIR}/output/smoke.npz" >/dev/null
    echo "smoke passed"

    python -m fomo_tune.cache_pooled --out "${POOLED}" --ckpt-path "${CKPT}"
else
    echo "pooled cache exists; skipping the GPU step"
fi

echo "=== grid ==="
python -m fomo_tune.bench_task7 \
    --cache "${POOLED}" \
    --out "${EXP_DIR}/output/grid.json" \
    | tee "${EXP_DIR}/output/grid.txt"

echo
echo "done. cache ${POOLED}, table ${EXP_DIR}/output/grid.txt"
echo "baseline to beat (mean + identity): MAE 3.50, r 0.968, age-bin spread 3.30"
