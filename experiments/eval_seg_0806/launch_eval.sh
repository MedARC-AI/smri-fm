#!/usr/bin/env bash
#SBATCH --job-name=eval_seg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=infinite
#SBATCH --partition=main
#SBATCH --output=slurms/slurm-%A_%a.out
#SBATCH --account=sophont
# #SBATCH --array=0-17
# #SBATCH --array=0-2
#SBATCH --array=3-17

set -euo pipefail

ROOT="/data/connor/nanobrain.1"
cd $ROOT

EXP_DIR="experiments/eval_seg_0806"
OUT_DIR="${EXP_DIR}/output"

# name, model, extra overrides. The name is the output-dir prefix, so sMRI MAE checkpoints
# stay distinguishable -- they all log `model: smri_mae`.
runs=(
    "random_features random_features"
    "random_unet     random_unet"
    "neurojepa       neurojepa"
    "neurovfm        neurovfm"
    "synthseg        synthseg"
    "smri_mae_vitl_fomo300 smri_mae model_kwargs.ckpt_path=/data/mihir-stuff/smri-pretrained/pretrain_full_90_10_h100/checkpoint-last.pth"
)

tasks=(
    fomo_task1_infarct_seg
    fomo_task2_meningioma
    fomo_task4_trigeminal
)

# One array element per (model, task) cell. This REQUIRES a warm HF dataset cache: the cells
# sharing a task start together and would otherwise race to build it.
total=$(( ${#runs[@]} * ${#tasks[@]} ))
if (( SLURM_ARRAY_TASK_ID >= total )); then
    echo "array index ${SLURM_ARRAY_TASK_ID} past the ${total}-cell grid; fix --array" >&2
    exit 1
fi

read -r name model extra <<<"${runs[$(( SLURM_ARRAY_TASK_ID / ${#tasks[@]} ))]}"
task=${tasks[$(( SLURM_ARRAY_TASK_ID % ${#tasks[@]} ))]}

if [[ -f "${OUT_DIR}/${name}__${task}/metrics.jsonl" ]]; then
    echo "result ${name}__${task} exists; skipping"
    exit 0
fi

echo "=== ${name}__${task} ==="
uv run --no-sync python -m nanobrain.eval.main "$model" "$task" \
    --overrides \
    output_root="${OUT_DIR}" \
    name="${name}__${task}" \
    ${extra}
