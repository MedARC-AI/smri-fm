#!/usr/bin/env bash
#SBATCH --job-name=task4_spatial
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --time=6:00:00
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

EXP_DIR="experiments/task4_spatial"
OUT_DIR="${EXP_DIR}/output"

WALNUT="hf://medarc/walnut/checkpoints/walnut-v0-1/vitl/sub-52k/checkpoint-last.pth"

N_PARALLEL=2

# name, train aug, test aug, alpha
runs=(
    "train-0_test-0_alpha-1e1 false false 1e1"
    "train-0_test-0_alpha-3.3 false false 3.3"
    "train-1_test-0_alpha-1e1 true  false 1e1"
    "train-0_test-1_alpha-1e1 false true  1e1"
    "train-1_test-1_alpha-1e1 true  true  1e1"
    "train-1_test-1_alpha-3e1 true  true  3e1"
)

run_one() {
    local name="$1" train="$2" tta="$3" alpha="$4"
    if [[ -f "${OUT_DIR}/${name}/metrics.json" ]]; then
        echo "result ${name} exists; skipping"
        return 0
    fi
    echo "=== ${name} ==="
    uv run --no-sync python -m fomo_tune.main_task4 train \
        output_root="${OUT_DIR}" \
        name="${name}" \
        ckpt_path="${WALNUT}" \
        head=logistic \
        alpha="${alpha}" \
        train_spatial="${train}" \
        tta_spatial="${tta}"
}
export -f run_one
export OUT_DIR WALNUT

printf '%s\n' "${runs[@]}" |
    parallel --will-cite --colsep ' +' --jobs "${N_PARALLEL}" --line-buffer --tagstring '{1}' \
        run_one {1} {2} {3} {4}
