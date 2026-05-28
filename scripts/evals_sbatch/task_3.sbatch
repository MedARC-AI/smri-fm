#!/usr/bin/env bash
#SBATCH --job-name=smri_fomo26_eval
#SBATCH --partition=main
#SBATCH --account=training
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=slurms/%x-%j.out
#SBATCH --error=slurms/%x-%j.err

set -euo pipefail

echo "Executing setup_asparagus_env.sh"
source scripts/setup_asparagus_env.sh
echo "Done loading environment variables"

echo "Job ID: ${SLURM_JOB_ID:-unset}"
echo "Node list: ${SLURM_JOB_NODELIST:-unset}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "ASPARAGUS_DATA: ${ASPARAGUS_DATA}"
echo "ASPARAGUS_MODELS: ${ASPARAGUS_MODELS}"


uv run asp_finetune_reg --config-name finetuning/20260525_bag_task_3.yaml

# Normalized labels to 0-1, I've had a custom dataset for that at: 
# uv run asp_finetune_reg --config-name finetuning/task_3__normalized_weights/20260526_bag_task_3_pool_mean.yaml
# uv run asp_finetune_reg --config-name finetuning/task_3__normalized_weights/20260526_bag_task_3_pool_cls.yaml

