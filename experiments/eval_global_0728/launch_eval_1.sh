#!/usr/bin/env bash
#SBATCH --job-name=eval_smri_mae
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=infinite
#SBATCH --partition=main
#SBATCH --output=slurms/slurm-%A_%a.out
#SBATCH --account=sophont
#SBATCH --array=0-14

set -euo pipefail

ROOT="/data/connor/nanobrain.1"
cd $ROOT

EXP_DIR="experiments/eval_global_0728"
OUT_DIR="${EXP_DIR}/output"

# suffix, checkpoint, transpose
runs=(
    "vitl_fomo300 /data/mihir-stuff/smri-pretrained/pretrain_full_90_10_h100/checkpoint-last.pth false"
    "vitb_12k     hf://mihirneal/walnut-v0-1/vitb/sub-12k/checkpoint-last.pth false"
    "vitb_52k     hf://mihirneal/walnut-v0-1/vitb/sub-52k/checkpoint-last.pth false"
    "vitl_12k     hf://mihirneal/walnut-v0-1/vitl/sub-12k/checkpoint-last.pth false"
    "vitl_52k     hf://mihirneal/walnut-v0-1/vitl/sub-52k/checkpoint-last.pth false"
    # the (Z, Y, X) axis order, against vitl_fomo300 above; one checkpoint is enough to see it
    "vitl_fomo300_tr /data/mihir-stuff/smri-pretrained/pretrain_full_90_10_h100/checkpoint-last.pth true"
)

# One array element per task, checkpoints looped inside: two runs on the same task would
# otherwise race to build the same HF dataset cache.
tasks=(
    abide_autism_control
    adhd200_adhd_control
    adni_ad_cn
    adni_age
    adni_amyloid_centiloid
    adni_tau_suvr
    cnp_adhd_control
    cnp_schz_bipolar_control
    dlbs_age
    fomo_task1_infarct
    fomo_task3_age
    fomo_task5_polymicrogyria
    ppmi_age
    ppmi_pd_cn
    ppmi_pd_prodromal
)

task=${tasks[$SLURM_ARRAY_TASK_ID]}

for run in "${runs[@]}"; do
    read -r suffix ckpt transpose <<<"$run"
    name="smri_mae_${suffix}__${task}"
    if [[ -f "${OUT_DIR}/${name}/metrics.jsonl" ]]; then
        echo "result ${name} exists; skipping"
        continue
    fi

    echo "=== ${name} ==="
    uv run --no-sync python -m nanobrain.eval.main smri_mae "$task" \
        --overrides \
        output_root="${OUT_DIR}" \
        name="${name}" \
        model_kwargs.ckpt_path="${ckpt}" \
        model_kwargs.global_pool=patch \
        model_kwargs.transpose="${transpose}"
done
