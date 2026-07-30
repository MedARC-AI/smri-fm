#!/bin/bash
#SBATCH --job-name=eval-adni-xyz
#SBATCH --partition=main
#SBATCH --account=training
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --array=0-37%24
#SBATCH --output=/admin/home/mihir.neal/smri-fm-adni-med/slurms/adni-mini-xyz-%A_%a.out
#SBATCH --error=/admin/home/mihir.neal/smri-fm-adni-med/slurms/adni-mini-xyz-%A_%a.err

set -euo pipefail

readonly WORKTREE=/admin/home/mihir.neal/smri-fm-adni-med
readonly CHECKPOINT_ROOT=/admin/home/mihir.neal/smri-fm/checkpoints
readonly PYTHON_BIN=/admin/home/mihir.neal/smri-fm/.venv/bin/python
readonly CACHE=/data/mihir-stuff/hf_cache_dir
readonly SNAPSHOT="${CACHE}/hub/datasets--medarc--adni-mini/snapshots/629f32de183020a76550d6b92c68ca539a5090cc"
readonly OUTPUT_ROOT="${WORKTREE}/output/eval_linear"
readonly TAG="${TAG:-adni_mini_xyz_pinned_wider_reg_fomolong_vitb_diversity_e099_20260730}"
readonly REPR=patch
readonly COMBOS_PER_TASK=2

cd "${WORKTREE}"
mkdir -p slurms "${OUTPUT_ROOT}"

export PYTHONPATH="${WORKTREE}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${CACHE}"
export HF_DATASETS_CACHE="${CACHE}"
export HF_HUB_CACHE="${CACHE}/hub"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export ADNI_MINI_SNAPSHOT="${SNAPSHOT}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

TASKS=(
  adni_age
  adni_sex
  adni_ad_cn
  adni_cn_mci_ad
  adni_synthseg_volumes
  adni_ad_cn_bag
  adni_mci_conversion_3y
  adni_cn_aplus_to_mci_ad_3y
  adni_adas13_slope_48m
  adni_ravlt_delayed_slope_48m
  adni_ravlt_learning_slope_48m
  adni_hippocampus_slope_48m
  adni_ventricle_slope_48m
  adni_entorhinal_thickness_slope_48m
  adni_cn_aplus_adas13_slope_48m
  adni_cn_aplus_ravlt_delayed_slope_48m
  adni_cn_aplus_ravlt_learning_slope_48m
  adni_cn_aplus_hippocampus_slope_48m
  adni_cn_aplus_ventricle_slope_48m
  adni_cn_aplus_entorhinal_thickness_slope_48m
  adni_amyloid_centiloid
  adni_tau_suvr
  adni_csf_abeta
  adni_csf_ptau
  adni_csf_ttau
)

CKPT_IDS=(
  fomolong_v1_vitb_n12000_e099
  fomolong_v1_vitb_n25000_e099
  fomolong_v1_vitb_n61051_e099
)

CKPTS=(
  "${CHECKPOINT_ROOT}/pretrain_fomo_long_v1_scale/scale_fomolong_v1_vitb_n12000_gbs256_s7338/checkpoint-00099.pth"
  "${CHECKPOINT_ROOT}/pretrain_fomo_long_v1_scale/scale_fomolong_v1_vitb_n25000_gbs256_s7338/checkpoint-00099.pth"
  "${CHECKPOINT_ROOT}/pretrain_fomo_long_v1_scale/scale_fomolong_v1_vitb_n61051_gbs256_s7338/checkpoint-00099.pth"
)

readonly NUM_TASKS=${#TASKS[@]}
readonly TOTAL=$((NUM_TASKS * ${#CKPTS[@]}))
readonly ARRAY_IDX=${SLURM_ARRAY_TASK_ID}
readonly COMBO_START=$((ARRAY_IDX * COMBOS_PER_TASK))

echo "Job ID:             ${SLURM_JOB_ID:-dry-run}"
echo "Array task:         ${ARRAY_IDX}"
echo "Total combinations: ${TOTAL}"
echo "Dataset:            medarc/adni-mini (split=eval)"
echo "Axis order:         canonical NIfTI XYZ; no X/Z permutation"

for ((OFFSET=0; OFFSET<COMBOS_PER_TASK; OFFSET++)); do
  COMBO=$((COMBO_START + OFFSET))
  if ((COMBO >= TOTAL)); then
    break
  fi

  TASK_IDX=$((COMBO % NUM_TASKS))
  CKPT_IDX=$((COMBO / NUM_TASKS))
  TASK=${TASKS[$TASK_IDX]}
  CKPT_ID=${CKPT_IDS[$CKPT_IDX]}
  CKPT=${CKPTS[$CKPT_IDX]}
  RUN_NAME="smri_mae__${TASK}__${REPR}__${TAG}__${CKPT_ID}"
  METRICS="${OUTPUT_ROOT}/${RUN_NAME}/metrics.json"

  echo "Combination:   ${COMBO}/${TOTAL}"
  echo "Task:          ${TASK}"
  echo "Checkpoint ID: ${CKPT_ID}"
  echo "Run name:      ${RUN_NAME}"

  if [[ "${ADNI_EVAL_DRY_RUN:-0}" == 1 ]]; then
    continue
  fi
  if [[ -s "${METRICS}" ]]; then
    echo "Metrics already exist; skipping: ${METRICS}"
    continue
  fi

  "${PYTHON_BIN}" scripts/run_adni_mini_linear.py smri_mae "${TASK}" \
    --overrides \
    name="${RUN_NAME}" \
    output_root="${OUTPUT_ROOT}" \
    model_kwargs.ckpt_path="${CKPT}" \
    model_kwargs.global_pool="${REPR}"
done
