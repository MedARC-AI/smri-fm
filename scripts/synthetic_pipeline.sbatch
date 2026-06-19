#!/bin/bash
#SBATCH --job-name=synthetic-pipeline
#SBATCH --partition=main
#SBATCH --account=training
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=slurms/%j.out
#SBATCH --error=slurms/%j.err

# Usage:
#   sbatch scripts/synthetic_pipeline.sbatch path/to/config.yaml [extra CLI flags]
#
# Example:
#   sbatch scripts/synthetic_pipeline.sbatch configs/synthetic.yaml --num-images 2

set -euo pipefail

CONFIG="${1:?Usage: sbatch scripts/synthetic_pipeline.sbatch path/to/config.yaml [extra CLI flags]}"
shift

mkdir -p slurms
export UV_CACHE_DIR="${UV_CACHE_DIR:-${SLURM_TMPDIR:-/tmp/$USER}/uv-cache}"
mkdir -p "$UV_CACHE_DIR"

echo "Config:     $CONFIG"
echo "GPUs:       ${SLURM_GPUS_PER_TASK:-1}"
echo "CPUs:       ${SLURM_CPUS_PER_TASK:-16}"
echo "Extra args: $*"
echo "==============================="

uv run python -m synthetic_pipeline.cli --config "$CONFIG" "$@"
