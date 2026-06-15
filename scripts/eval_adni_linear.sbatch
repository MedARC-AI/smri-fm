#!/bin/bash
#SBATCH --job-name=eval_adni_linear
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

CKPT=/data/mihir-stuff/smri-pretrained/pretrain_full_90_10_h100/checkpoint-last.pth
CACHE=/data/mihir-stuff/hf_cache_dir
export HF_HUB_CACHE="${CACHE}/hub"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
DATASET="${1:-adni_sex}"   # adni_sex | adni_ad_cn | adni_age | adni_synthseg_volumes
REPR="${2:-cls}"           # cls | patch

echo "Dataset:          ${DATASET}"
echo "Representation:   ${REPR}"
uv run python -m eval.main_linear smri_mae "${REPR}" "${DATASET}" \
    --overrides \
    model_kwargs.ckpt_path="${CKPT}" \
    dataset_kwargs.cache_dir="${CACHE}" \
    map_workers=8 \
    num_workers=8
