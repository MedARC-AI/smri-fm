#!/usr/bin/env bash
# Build submission for a FOMO26 task. 
# The submission is a self-contained Apptainer image (.sif) that runs inference for that task.
#
# Usage:
#   scripts/build_fomo26_submission.sh <task> <config.yaml> <checkpoint.pth> <output.sif>
#
# <task>                the task number, ie 1,...,7.
# <model_config.yaml>   the config.yaml, as produced by the hydra run
# <checkpoint.pth>      the path to the finetuned checkpoint (for tasks 1-5) or the pretrained model (for tasks 6-7)
# <output.sif>          the output location
#
# Example:
#   scripts/build_fomo26_submission.sh 6 path/to/task6/config.yaml path/to/task6/ckpt-best.pth  out/task6.sif

set -euo pipefail

task=$1 cfg=$(realpath "$2") ckpt=$(realpath "$3") sif=$(realpath -m "$4")
repo=$(realpath "$(dirname "$0")/..")

# create temporary build dir, and remove it after script run
build=$(mktemp -d)
trap 'rm -rf "$build"' EXIT

# create image definition
cat > "$build/Apptainer.def" <<EOF
Bootstrap: docker
From: pytorch/pytorch:2.12.1-cuda13.2-cudnn9-runtime

%environment
    export PYTHONUNBUFFERED=1 LC_ALL=C.UTF-8 FOMO_TASK=$task

%files
    $cfg                                  /app/config.yaml
    $ckpt                                 /app/checkpoint.pth
    pyproject.toml                        /app/pyproject.toml
    scripts/predict_fomo26.py             /app/predict.py
    scripts/preprocessing_fomo26.py       /app/preprocessing.py
    src/asparagus_bridge                  /app/src/asparagus_bridge
    src/evaluation                        /app/src/evaluation
    src/smri_mae                          /app/src/smri_mae
    third_party/asparagus                 /app/third_party/asparagus
    third_party/asparagus_preprocessing   /app/third_party/asparagus_preprocessing

%post
    mkdir -p /input /output
    pip install --no-cache-dir --break-system-packages -U pip
    pip install --no-cache-dir --break-system-packages uv
    cd /app && SETUPTOOLS_SCM_PRETEND_VERSION=1.0.0 uv pip install --system --break-system-packages .
    chmod +x /app/predict.py

%runscript
    exec python /app/predict.py "\$@"
EOF

# build image
mkdir -p "$(dirname "$sif")"
cd "$repo" && apptainer build --arch amd64 --fakeroot "$sif" "$build/Apptainer.def"
