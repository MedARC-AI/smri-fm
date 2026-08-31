#!/usr/bin/env bash
#SBATCH --job-name=fomo_tune_v3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --time=3:00:00
#SBATCH --partition=main
#SBATCH --array=4
#SBATCH --output=slurms/slurm-%A_%a.out
#SBATCH --account=sophont
#SBATCH --qos=high

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd $ROOT

set -a
source .env
set +a

EXP_DIR="experiments/fomo_tune_v3"
OUT_DIR="${EXP_DIR}/output"

runs=(
    "task1 main_task1"
    "task5 main_task5"
    "task3 main_task3"
    "task2 main_task2"
    "task4 main_task4"
)

read -r name module <<<"${runs[${SLURM_ARRAY_TASK_ID}]}"

if [[ -f "${OUT_DIR}/${name}/metrics.json" ]]; then
    echo "result ${name} exists; skipping"
    exit 0
fi

echo "=== ${name} ==="
uv run --no-sync python -m "fomo_tune.${module}" train \
    output_root="${OUT_DIR}" \
    name="${name}"

cat "${OUT_DIR}/${name}/metrics.json"
