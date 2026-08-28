#!/usr/bin/env bash
#SBATCH --job-name=task4_logistic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --time=3:00:00
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

EXP_DIR="experiments/task4_logistic"
OUT_DIR="${EXP_DIR}/output"

PT_FULL="hf://medarc/walnut/checkpoints/pretrain_full_90_10_h100/checkpoint-last.pth"
WALNUT="hf://medarc/walnut/checkpoints/walnut-v0-1/vitl/sub-52k/checkpoint-last.pth"

# `decode` holds ~19GB a process; four at once thrashed reclaim on a shared node
N_PARALLEL=2

runs=(
    "ridge_ptfull        ridge    ${PT_FULL} 0"
    "logistic_ptfull_1e4 logistic ${PT_FULL} 1e4"
    "ridge_walnut        ridge    ${WALNUT}  0"
    "logistic_walnut_1e4 logistic ${WALNUT}  1e4"
    "logistic_ptfull_1e2 logistic ${PT_FULL} 1e2"
    "logistic_ptfull_1e6 logistic ${PT_FULL} 1e6"
    "logistic_walnut_1e2 logistic ${WALNUT}  1e2"
    "logistic_walnut_1e6 logistic ${WALNUT}  1e6"
)

run_one() {
    local name="$1" head="$2" ckpt="$3" alpha="$4"
    if [[ -f "${OUT_DIR}/${name}/metrics.json" ]]; then
        echo "result ${name} exists; skipping"
        return 0
    fi
    echo "=== ${name} ==="
    uv run --no-sync python -m fomo_tune.main_task4 train \
        output_root="${OUT_DIR}" \
        name="${name}" \
        ckpt_path="${ckpt}" \
        head="${head}" \
        alpha="${alpha}"
}
export -f run_one
export OUT_DIR

printf '%s\n' "${runs[@]}" |
    parallel --will-cite --colsep ' +' --jobs "${N_PARALLEL}" --line-buffer --tagstring '{1}' \
        run_one {1} {2} {3} {4}
