#!/bin/bash
#SBATCH --job-name=synthseg
#SBATCH --partition=main
#SBATCH --account=training
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=slurms/%j.out
#SBATCH --error=slurms/%j.err
#SBATCH --container-image=freesurfer/freesurfer:7.4.1
#SBATCH --container-mounts=/admin/home/rdatchane/smri-fm:/workspace

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
echo "Flags:  --parc $EXTRA_FLAGS"
echo "==============================="

for img in "$INPUT_DIR"/*.nii.gz; do
    base=$(basename "$img" .nii.gz)
    echo "Processing: $base"
    mri_synthseg \
        --i "$img" \
        --o "$OUTPUT_DIR/${base}_synthseg.nii.gz" \
        --vol "$OUTPUT_DIR/${base}_volumes.csv" \
        --qc "$OUTPUT_DIR/${base}_qc.csv" \
        --parc \
        $EXTRA_FLAGS
done

echo "=== Done ==="
