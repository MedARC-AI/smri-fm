#!/usr/bin/env bash
#SBATCH --job-name=fomo_tune
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=2:00:00
#SBATCH --partition=main
#SBATCH --output=slurms/slurm-%j.out
#SBATCH --account=sophont

set -euo pipefail

ROOT="/data/connor/fomo_tune"
cd $ROOT

EXP_DIR="experiments/fomo_tune_baseline"
OUT_DIR="${EXP_DIR}/output"

# name, module. Cheapest first, so a broken environment fails in 90s.
# One job, run sequentially: the whole grid is ~12 minutes and each task loads the same
# 3.9G checkpoint, so an array would only add queue time and cache contention.
runs=(
    "task1 main_task1"
    "task5 main_task5"
    "task3 main_task3"
    "task2 main_task2"
)

for run in "${runs[@]}"; do
    read -r name module <<<"${run}"

    if [[ -f "${OUT_DIR}/${name}/metrics.json" ]]; then
        echo "result ${name} exists; skipping"
        continue
    fi

    echo "=== ${name} ==="
    uv run --no-sync python -m "fomo_tune.${module}" train \
        output_root="${OUT_DIR}" \
        name="${name}"
done

echo "=== results ==="
cat "${OUT_DIR}"/*/metrics.json
