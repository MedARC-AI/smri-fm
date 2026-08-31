#!/usr/bin/env bash

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd $ROOT

EXP_DIR="experiments/fomo_tune_v3"
OUT_DIR="${EXP_DIR}/output"

name=task6_and_7

uv run --no-sync python -m fomo_tune.main_task6_and_7 export \
    output_root="${OUT_DIR}" \
    name="${name}" \
    device=cpu
