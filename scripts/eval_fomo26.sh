#!/usr/bin/env bash
# Convert a smri-fm pretrain checkpoint and run asparagus finetuning on the
# FOMO26 downstream tasks.
#
# Usage:
#   scripts/eval_fomo26.sh <model_name> <pretrain_checkpoint.pth>
#
# <model_name> selects both the asparagus +model= overlay and the checkpoint
# converter registered in asparagus_bridge.checkpoint.CONVERTERS.
#
# Task lists (whitespace-separated task names matching asparagus task configs):
#   FOMO_SEG_TASKS   default empty
#   FOMO_CLS_TASKS   default empty (set once FOMO26 cls task IDs are pinned)
#   FOMO_REG_TASKS   default empty (set once FOMO26 reg task IDs are pinned)
#
# Example:
#   FOMO_CLS_TASKS="DEBUG_FT_CLS" scripts/eval_fomo26.sh smri_mae runs/mae/checkpoint-last.pth

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: $(basename "$0") <model_name> <pretrain_checkpoint.pth>" >&2
    exit 2
fi

model_name="$1"
src_ckpt="$2"

repo="$(git rev-parse --show-toplevel)"
source "$repo/scripts/setup_asparagus_env.sh"

asparagus_ckpt="${src_ckpt%.pth}.asparagus.ckpt"
echo ">>> converting checkpoint ($model_name): $src_ckpt -> $asparagus_ckpt"
python -c "
from asparagus_bridge.checkpoint import convert_checkpoint
convert_checkpoint('$model_name', '$src_ckpt', '$asparagus_ckpt')
"

run_finetune() {
    local cmd="$1" task="$2"
    echo ">>> $cmd task=$task"
    "$cmd" task="$task" +model="$model_name" checkpoint_path="$asparagus_ckpt"
}

for task in ${FOMO_SEG_TASKS:-}; do run_finetune asp_finetune_seg "$task"; done
for task in ${FOMO_CLS_TASKS:-}; do run_finetune asp_finetune_cls "$task"; done
for task in ${FOMO_REG_TASKS:-}; do run_finetune asp_finetune_reg "$task"; done
