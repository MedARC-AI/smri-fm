#!/usr/bin/env bash
# Convert a smri_mae pretrain checkpoint and run asparagus finetuning on the
# FOMO26 downstream tasks. Bypasses asp_eval_box_run because it only forwards
# checkpoint_run_id (no path passthrough); we call asp_finetune_* directly.
#
# Usage:
#   scripts/eval_fomo26.sh <pretrain_checkpoint.pth>
#
# Task lists (whitespace-separated task names matching asparagus task configs):
#   FOMO_SEG_TASKS   default empty (SmriMaeSegBackbone is not yet implemented)
#   FOMO_CLS_TASKS   default empty (set once FOMO26 cls task IDs are pinned)
#   FOMO_REG_TASKS   default empty (set once FOMO26 reg task IDs are pinned)
#
# Example:
#   FOMO_CLS_TASKS="DEBUG_FT_CLS" scripts/eval_fomo26.sh runs/mae/checkpoint-last.pth

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $(basename "$0") <pretrain_checkpoint.pth>" >&2
    exit 2
fi

src_ckpt="$1"

repo="$(git rev-parse --show-toplevel)"
# shellcheck source=setup_asparagus_env.sh
source "$repo/scripts/setup_asparagus_env.sh"

asparagus_ckpt="${src_ckpt%.pth}.asparagus.ckpt"
echo ">>> converting checkpoint: $src_ckpt -> $asparagus_ckpt"
python -c "
from asparagus_bridge.checkpoint import convert_smri_mae_checkpoint
convert_smri_mae_checkpoint('$src_ckpt', '$asparagus_ckpt')
"

run_finetune() {
    local cmd="$1" task="$2"
    echo ">>> $cmd task=$task"
    "$cmd" task="$task" +model=smri_mae checkpoint_path="$asparagus_ckpt"
}

for task in ${FOMO_SEG_TASKS:-}; do run_finetune asp_finetune_seg "$task"; done
for task in ${FOMO_CLS_TASKS:-}; do run_finetune asp_finetune_cls "$task"; done
for task in ${FOMO_REG_TASKS:-}; do run_finetune asp_finetune_reg "$task"; done
