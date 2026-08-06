#!/usr/bin/env bash
#SBATCH --job-name=eval_global
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=infinite
#SBATCH --partition=main
#SBATCH --output=slurms/slurm-%A_%a.out
#SBATCH --account=sophont
# #SBATCH --array=0-14
#SBATCH --array=15-17

set -euo pipefail

ROOT="/data/connor/nanobrain.1"
cd $ROOT

EXP_DIR="experiments/eval_global_0728"
OUT_DIR="${EXP_DIR}/output"

models=(
    random_features
    random_unet
    neurojepa
    neurovfm
    synthseg
)

# One array element per task, models looped inside: two models on the same task would
# otherwise race to build the same HF dataset cache.
tasks=(
    abide_autism_control
    adhd200_adhd_control
    adni_ad_cn
    adni_age
    adni_amyloid_centiloid
    adni_sex
    adni_tau_suvr
    cnp_adhd_control
    cnp_schz_bipolar_control
    cnp_sex
    dlbs_age
    dlbs_sex
    fomo_task1_infarct
    fomo_task3_age
    fomo_task5_polymicrogyria
    ppmi_age
    ppmi_pd_cn
    ppmi_pd_prodromal
)

task=${tasks[$SLURM_ARRAY_TASK_ID]}

for model in "${models[@]}"; do
    name="${model}__${task}"
    if [[ -f "${OUT_DIR}/${name}/metrics.jsonl" ]]; then
        echo "result ${name} exists; skipping"
        continue
    fi

    echo "=== ${name} ==="
    uv run --no-sync python -m nanobrain.eval.main "$model" "$task" \
        --overrides \
        output_root="${OUT_DIR}" \
        name="${name}"
done
