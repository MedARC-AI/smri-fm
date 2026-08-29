#!/usr/bin/env bash
#SBATCH --job-name=task5_cortex
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --time=2:00:00
#SBATCH --partition=main
#SBATCH --output=slurms/slurm-%j.out
#SBATCH --account=sophont
#SBATCH --qos=high

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd $ROOT

set -a
source .env
set +a

EXP_DIR="experiments/task5_cortex"
OUT_DIR="${EXP_DIR}/output"

N_PARALLEL=3

PT_FULL="hf://medarc/walnut/checkpoints/pretrain_full_90_10_h100/checkpoint-last.pth"
WALNUT="hf://medarc/walnut/checkpoints/walnut-v0-1/vitl/sub-52k/checkpoint-last.pth"

# cortex_frac is the fraction of a patch that must be cortex for the patch to be pooled.
# 0.0 is the any()-voxel mask.
# name, ckpt, pooling, cortex_frac
runs=(
    "ckpt-ptfull_pool-global ${PT_FULL} global 0.0"
    "ckpt-ptfull_pool-cortex000 ${PT_FULL} cortex 0.0"
    "ckpt-ptfull_pool-cortex010 ${PT_FULL} cortex 0.1"
    "ckpt-ptfull_pool-cortex025 ${PT_FULL} cortex 0.25"
    "ckpt-walnut_pool-global ${WALNUT} global 0.0"
    "ckpt-walnut_pool-cortex000 ${WALNUT} cortex 0.0"
    "ckpt-walnut_pool-cortex010 ${WALNUT} cortex 0.1"
    "ckpt-walnut_pool-cortex025 ${WALNUT} cortex 0.25"
)

run_one() {
    local name="$1" ckpt="$2" pooling="$3" cortex_frac="$4"
    if [[ -f "${OUT_DIR}/${name}/metrics.json" ]]; then
        echo "result ${name} exists; skipping"
        return 0
    fi
    echo "=== ${name} ==="
    uv run --no-sync python -m fomo_tune.main_task5 train \
        output_root="${OUT_DIR}" \
        ckpt_path="${ckpt}" \
        name="${name}" \
        pooling="${pooling}" \
        cortex_frac="${cortex_frac}"
}
export -f run_one
export OUT_DIR

printf '%s\n' "${runs[@]}" |
    parallel --will-cite --colsep ' ' --jobs "${N_PARALLEL}" --line-buffer --tagstring '{1}' \
        run_one {1} {2} {3} {4}
