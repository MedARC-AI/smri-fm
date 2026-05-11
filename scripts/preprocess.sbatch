#!/bin/bash
#SBATCH --job-name=preprocess
#SBATCH --partition=main
#SBATCH --account=training
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --output=slurms/%j.out
#SBATCH --error=slurms/%j.err

# Usage:
#   sbatch scripts/preprocess.sbatch <input_dir> [extra pipeline flags]
#
# Examples:
#   sbatch scripts/preprocess.sbatch /path/to/dataset
#   sbatch scripts/preprocess.sbatch /path/to/dataset --cpu
#   sbatch scripts/preprocess.sbatch /path/to/dataset --batch-id 0 --batch-size 100

set -euo pipefail

INPUT="${1:?Usage: sbatch scripts/preprocess.sbatch <input_dir> [extra pipeline flags]}"
shift

mkdir -p slurms
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

SYNTHSEG_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "Input:            $INPUT"
echo "SynthSeg threads: $SYNTHSEG_THREADS"
echo "GPUs:             ${SLURM_GPUS_PER_TASK:-0}"
echo "Extra args:       $*"
echo "==============================="

uv run python src/preprocessing/pipeline.py \
    --input "$INPUT" \
    --synthseg-threads "$SYNTHSEG_THREADS" \
    "$@"
