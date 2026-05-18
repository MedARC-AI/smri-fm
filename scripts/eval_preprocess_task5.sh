#!/usr/bin/env bash
# Extract and preprocess FOMO26 Task 5 into asparagus format.
#
# Usage:
#   scripts/eval_preprocess_task5.sh
#
# Optional environment variables:
#   TASK5_NUM_WORKERS   number of workers for asp_process (default: 4)
#   TASK5_EXTRACT_FLAGS extra flags for Task_5_extract.py (default: --verbose)

set -euo pipefail

repo="$(git rev-parse --show-toplevel)"
source "$repo/scripts/setup_asparagus_env.sh"

task5_script="$ASPARAGUS_SOURCE/Task_5_extract.py"
task5_zip="$ASPARAGUS_SOURCE/Zhang_Lingfeng_2022_PPMR_Dataset.zip"
num_workers="${TASK5_NUM_WORKERS:-4}"
extract_flags="${TASK5_EXTRACT_FLAGS:---verbose}"

if [[ ! -f "$task5_script" ]]; then
    echo "Missing Task 5 extractor: $task5_script" >&2
    exit 1
fi

if [[ ! -f "$task5_zip" ]]; then
    echo "Missing Task 5 dataset zip: $task5_zip" >&2
    exit 1
fi

echo ">>> extracting Task 5 source data in $ASPARAGUS_SOURCE"
(
    cd "$ASPARAGUS_SOURCE"
    # shellcheck disable=SC2086
    uv run --project "$repo" python Task_5_extract.py $extract_flags
)

echo ">>> preprocessing Task 5 as CLS003_FOMO26_Polymicrogyria"
uv run --project "$repo" asp_process \
    --dataset CLS003 \
    --save_as_tensor \
    --num_workers "$num_workers"

echo ">>> creating Task 5 80/10/10 split"
uv run --project "$repo" asp_split \
    --dataset CLS003_FOMO26_Polymicrogyria \
    --vals 80 10 10
