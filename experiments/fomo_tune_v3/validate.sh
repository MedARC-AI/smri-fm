#!/usr/bin/env bash
#SBATCH --job-name=fomo_tune_v3_validate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --time=2:00:00
#SBATCH --partition=main
#SBATCH --nodelist=n-4,n-6,n-7
#SBATCH --output=slurms/slurm-%j.out
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

if ! command -v apptainer > /dev/null; then
    echo "no apptainer on $(hostname); resubmit, or ask for an install on this node"
    exit 1
fi

# runs=(task1 task5 task3)
# runs=(task2 task4)
runs=(task6_and_7)

results=()
status=0

for name in "${runs[@]}"; do
    sif="${OUT_DIR}/${name}/${name}.sif"

    if [[ ! -f "${sif}" ]]; then
        echo "sif ${sif} missing; skipping"
        results+=("SKIP  ${name}")
        continue
    fi

    echo "=== ${name} ==="
    if uv run --no-sync python -m fomo_tune.validate "${OUT_DIR}/${name}"; then
        results+=("PASS  ${name}")
    else
        results+=("FAIL  ${name}")
        status=1
    fi
done

echo "=== summary ==="
printf '%s\n' "${results[@]}"
exit ${status}
