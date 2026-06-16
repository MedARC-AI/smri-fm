#!/bin/bash
#SBATCH --job-name=eval_adni_sweep
#SBATCH --partition=main
#SBATCH --account=training
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --output=slurms/%j.out
#SBATCH --error=slurms/%j.err

set -euo pipefail

mkdir -p slurms
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
# Keep BLAS/OpenMP single-threaded so the dataset.map fork doesn't deadlock.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "Job ID:           ${SLURM_JOB_ID:-unknown}"
echo "Node:             ${SLURM_NODELIST:-unknown}"
echo "==============================="

CACHE=/data/mihir-stuff/hf_cache_dir
# Send the hub download cache to the same dir (default is ~/.cache on the home partition).
export HF_HUB_CACHE="${CACHE}/hub"
CKPT_ROOT=/data/mihir-stuff/smri-pretrained
OUT=output/eval_sweep/adni_checkpoint_sweep.csv

# all checkpoints, both runs, sorted (run, epoch)
mapfile -t CKPTS < <(ls "${CKPT_ROOT}"/*/checkpoint-*.pth | sort)
echo "Sweeping ${#CKPTS[@]} checkpoints:"
printf '  %s\n' "${CKPTS[@]}"

uv run python -m eval.sweep_checkpoints "${CKPTS[@]}" \
    --cache_dir "${CACHE}" \
    --out "${OUT}" \
    --overrides map_workers=8 num_workers=8
