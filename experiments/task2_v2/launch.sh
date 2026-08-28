#!/usr/bin/env bash
#SBATCH --job-name=task2_v2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --time=2:00:00
#SBATCH --partition=main
#SBATCH --array=2-3
#SBATCH --output=slurms/slurm-%A_%a.out
#SBATCH --account=sophont
#SBATCH --qos=high

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd $ROOT

set -a
source .env
set +a

EXP_DIR="experiments/task2_v2"
OUT_DIR="${EXP_DIR}/output"

PT_FULL="hf://medarc/walnut/checkpoints/pretrain_full_90_10_h100/checkpoint-last.pth"
WALNUT="hf://medarc/walnut/checkpoints/walnut-v0-1/vitl/sub-52k/checkpoint-last.pth"

# One run per checkpoint, at the defaults. The two scored 0.195 each under the baseline head, so
# this is the first read on whether the progressive head prefers one of them.
#
# 0 and 1 are job 395096, launched before `leave_one_out` saved per-fold predictions. 2 and 3
# repeat them at the same seeds with the predictions, so the array starts at 2 while 0 and 1 are
# still in flight and the guard below cannot see them. Widen back to 0-3 once they have finished.
ckpts=("${PT_FULL}" "${WALNUT}" "${PT_FULL}" "${WALNUT}")
names=(ckpt-ptfull ckpt-walnut ckpt-ptfull_folds ckpt-walnut_folds)

ckpt="${ckpts[${SLURM_ARRAY_TASK_ID}]}"
name="${names[${SLURM_ARRAY_TASK_ID}]}"

if [[ -f "${OUT_DIR}/${name}/metrics.json" ]]; then
    echo "result ${name} exists; skipping"
    exit 0
fi

echo "=== ${name} ==="
uv run --no-sync python -m fomo_tune.main_task2 train \
    output_root="${OUT_DIR}" \
    ckpt_path="${ckpt}" \
    name="${name}"
