#!/usr/bin/env bash
# Package each trained run into its challenge .sif. Run on the login node, after launch.sh:
# apptainer lives there and on a couple of compute nodes only, and a build needs no GPU driver.
#
# Slow. Apptainer always re-runs %post, so each run re-downloads ~3G of wheels.

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd $ROOT

set -a
source .env
set +a

EXP_DIR="experiments/fomo_tune_v3"
OUT_DIR="${EXP_DIR}/output"

runs=(task1 task5 task3 task2 task4 task6_and_7 )

for name in "${runs[@]}"; do
    # build.py names the sif after `task` in the run's saved config, which is the run name here
    sif="${OUT_DIR}/${name}/${name}.sif"

    if [[ -f "${sif}" ]]; then
        echo "sif ${sif} exists; skipping"
        continue
    fi

    echo "=== ${name} ==="
    uv run --no-sync python -m fomo_tune.build "${OUT_DIR}/${name}"
done

echo "=== sifs ==="
ls -lh "${OUT_DIR}"/*/*.sif
