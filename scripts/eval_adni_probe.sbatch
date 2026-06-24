#!/bin/bash
#SBATCH --job-name=eval_adni_probe
#SBATCH --partition=main
#SBATCH --account=training
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=4:00:00
#SBATCH --output=slurms/%j.out
#SBATCH --error=slurms/%j.err

set -euo pipefail

mkdir -p slurms

echo "Job ID:           ${SLURM_JOB_ID:-unknown}"
echo "Node:             ${SLURM_NODELIST:-unknown}"
echo "GPUs:             ${SLURM_GPUS:-${SLURM_GPUS_PER_TASK:-unknown}}"
echo "CPUs per task:    ${SLURM_CPUS_PER_TASK:-unknown}"
echo "==============================="

DEFAULT_CKPT=/data/mihir-stuff/smri-pretrained/pretrain_full_90_10_h100/checkpoint-last.pth
CACHE=/data/mihir-stuff/hf_cache_dir
export HF_DATASETS_CACHE="${CACHE}"
export HF_HUB_CACHE="${CACHE}/hub"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

TASK="${1:-adni_sex}"          # adni_sex | adni_age | adni_ad_cn | adni_cn_mci_ad | adni_synthseg_volumes | adni_ad_cn_bag
REPR="${2:-cls}"              # cls | patch
RUN_TAG="${3:-$(date -u +%Y%m%d_%H%M%S)}"
CKPT="${4:-${DEFAULT_CKPT}}"
RUN_NAME="smri_mae__${TASK}__${REPR}__${RUN_TAG}"

echo "Task:             ${TASK}"
echo "Representation:   ${REPR}"
echo "Checkpoint:       ${CKPT}"
echo "Run name:         ${RUN_NAME}"
uv run python -m evaluation.main_linear smri_mae "${TASK}" \
    --overrides \
    name="${RUN_NAME}" \
    model_kwargs.ckpt_path="${CKPT}" \
    model_kwargs.global_pool="${REPR}"
