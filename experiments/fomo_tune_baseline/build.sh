#!/usr/bin/env bash
# Package each trained run into its challenge .sif. Run on the login node, after launch.sh:
# apptainer lives there and on n-6 only, and a build needs no GPU driver.
#
# Slow. Apptainer always re-runs %post, so each of the three re-downloads ~3G of wheels.

set -euo pipefail

ROOT="/data/connor/fomo_tune"
cd $ROOT

EXP_DIR="experiments/fomo_tune_baseline"
OUT_DIR="${EXP_DIR}/output"

runs=(task1 task5 task3)

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
