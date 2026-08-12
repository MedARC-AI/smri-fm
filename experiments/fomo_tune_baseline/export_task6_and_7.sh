#!/usr/bin/env bash
# Tasks 6 and 7: write the run dir, then package it. Both steps run on the login node -- nothing
# is fitted, so there is no counterpart in launch.sh and no GPU anywhere in this script.
#
# device=cpu is what makes the export work here: it only constructs the backbone, and the login
# node has no driver to move it to. `predict` picks its own device inside the container.
#
# Validate the result with:
#   uv run python third_party/container-validator/container_validator/validate.py \
#       --task task6_and_7 --sif experiments/fomo_tune_baseline/output/task6_and_7/task6_and_7.sif

set -euo pipefail

ROOT="/data/connor/fomo_tune"
cd $ROOT

EXP_DIR="experiments/fomo_tune_baseline"
OUT_DIR="${EXP_DIR}/output"

name=task6_and_7

uv run --no-sync python -m fomo_tune.main_task6_and_7 export \
    output_root="${OUT_DIR}" \
    name="${name}" \
    device=cpu

uv run --no-sync python -m fomo_tune.build "${OUT_DIR}/${name}"

ls -lh "${OUT_DIR}/${name}/${name}.sif"
