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

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a

  ENV_KEYS=()
  while IFS='=' read -r key _; do
    [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    [[ -n "${!key:-}" ]] || continue
    ENV_KEYS+=("$key")
  done < <(grep -E '^[A-Za-z_][A-Za-z0-9_]*=' .env || true)

  if ((${#ENV_KEYS[@]} > 0)); then
    echo "Loaded .env keys: ${ENV_KEYS[*]}"
  else
    echo "Loaded .env, but no non-empty keys were set."
  fi
else
  echo "No .env file found; using existing environment only."
fi

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
