#!/usr/bin/env bash
# Build submission for a FOMO26 task. 
# The submission is a self-contained Apptainer image (.sif) that runs inference for that task.
#
# Usage:
#   scripts/build_fomo26_submission.sh <task> <checkpoint.pth> <output.sif>
#
# <task>                the task number, ie 1,...,7.
# <model_config.yaml>   the config.yaml, as produced by the hydra run
# <checkpoint.pth>      the path to the finetuned checkpoint (for tasks 1-5) or the pretrained model (for tasks 6-7)
# <output.sif>          the output location
#
# Example:
#   scripts/build_fomo26_submission.sh 6 path/to/task6/ckpt-best.pth out/task6.sif

set -euo pipefail

task=$1 cfg=$2 ckpt=$3 sif=$4

sif=$(cd "$(dirname "$sif")" && pwd)/$(basename "$sif") 
cfg=$(cd "$(dirname "$cfg")" && pwd)/$(basename "$cfg") 
ckpt=$(cd "$(dirname "$ckpt")" && pwd)/$(basename "$ckpt")
repo=$(cd "$(dirname "$0")/.." && pwd)

# select prediction file based on task
if [ "$task" = 6 ] || [ "$task" = 7 ]; then tmpl=predict_task_6_7.py
else tmpl=predict_task_$task.py; fi

# create temporary build dir, and remove it after script run
build=$(mktemp -d)
trap 'rm -rf "$build"' EXIT

# make copy of pyproject.toml with fixed version number, as fix version number in pyproject.toml, as we can't use git history for versioning
sed 's/dynamic = \["version"\]/version = "1.0.0"/' "$repo/pyproject.toml" > "$build/pyproject.toml"

# create image defnition
cat > "$build/Apptainer.def" <<EOF
Bootstrap: docker
From: pytorch/pytorch:2.12.1-cuda13.2-cudnn9-runtime

%environment
    export PYTHONUNBUFFERED=1
    export LC_ALL=C.UTF-8

%files
    $repo/scripts/submission_templates/$tmpl      /app/predict.py
    $build/pyproject.toml                         /app/pyproject.toml
    $cfg                                          /app/config.yaml
    $ckpt                                         /app/checkpoint.pth
    $repo/src/asparagus_bridge                    /app/src/asparagus_bridge
    $repo/src/evaluation                          /app/src/evaluation
    $repo/src/smri_mae                            /app/src/smri_mae
    $repo/third_party/asparagus                   /app/third_party/asparagus
    $repo/third_party/asparagus_preprocessing     /app/third_party/asparagus_preprocessing

%post
    mkdir -p /input /output
    pip install --no-cache-dir --break-system-packages -U pip
    pip install --no-cache-dir --break-system-packages uv
    cd /app && uv pip install --system --break-system-packages .
    chmod +x /app/predict.py

%runscript
    exec python /app/predict.py "\$@"
EOF

# build image
mkdir -p "$(dirname "$sif")"
cd $build
apptainer build --arch amd64 --fakeroot "$sif" Apptainer.def 
