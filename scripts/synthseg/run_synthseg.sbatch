#!/bin/bash
#SBATCH --job-name=synthseg
#SBATCH --partition=main
#SBATCH --account=training
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=slurms/%j.out
#SBATCH --error=slurms/%j.err
#SBATCH --container-image=freesurfer/freesurfer:7.4.1
#SBATCH --container-mounts=/admin/home/rdatchane/smri-fm:/workspace

# NOTE: FreeSurfer 7.4.1 Docker image is CPU-only (no CUDA/GPU support).
# Benchmarked at ~29s/scan with 8 threads vs ~130s/scan with 1 thread.
# No GPU requested — avoids wasting an H100 slot.

# Usage:
#   sbatch scripts/synthseg/run_synthseg.sbatch <input_dir> <output_dir> [--robust]
#
# Examples:
#   sbatch scripts/synthseg/run_synthseg.sbatch datasets/DLBS/data/raw datasets/DLBS/data/synthseg
#   sbatch scripts/synthseg/run_synthseg.sbatch datasets/DLBS/data/raw datasets/DLBS/data/synthseg_robust --robust

INPUT_DIR="/workspace/${1:?Usage: sbatch run_synthseg.sbatch <input_dir> <output_dir> [--robust]}"
OUTPUT_DIR="/workspace/${2:?Usage: sbatch run_synthseg.sbatch <input_dir> <output_dir> [--robust]}"
EXTRA_FLAGS="${3:-}"

mkdir -p "$OUTPUT_DIR"

echo "Input:  $INPUT_DIR"
echo "Output: $OUTPUT_DIR"
echo "Flags:  --parc --threads 8 $EXTRA_FLAGS"
echo "==============================="

# Use folder input so the model is loaded once (not per scan).
# Use --threads 8 for ~4.5x speedup over single-threaded.
mri_synthseg \
    --i "$INPUT_DIR" \
    --o "$OUTPUT_DIR" \
    --vol "$OUTPUT_DIR/volumes.csv" \
    --qc "$OUTPUT_DIR/qc.csv" \
    --threads 8 \
    --parc \
    $EXTRA_FLAGS

echo "=== Done ==="
