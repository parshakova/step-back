#!/bin/bash
#
# Run all experiments from cifar10_resnet20_compare_seeds.json in parallel.
#
# ResNet-20 (0.27M params) on CIFAR-10, 50 epochs, 21 opt configs x 2 seeds = 42 tasks.
# ~30 min per job, 8GB per GPU.
#
# Usage:
#   ./slurm_scripts/run_cifar10_resnet20_compare_seeds.sh

set -e

CONFIG_ID="cifar10_resnet20_compare_seeds"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs/${CONFIG_ID}"
RUN_LOGS="${LOG_DIR}/run_logs"
TASK_FILE="${LOG_DIR}/tasks"

mkdir -p "$LOG_DIR" "$RUN_LOGS" "${ROOT_DIR}/slurm_logs" "${ROOT_DIR}/disbatch_logs" "${ROOT_DIR}/output"

echo "=== ${CONFIG_ID} ==="
echo "Generating task file..."

cd "$ROOT_DIR"
python slurm_scripts/generate_tasks.py \
    --config_id="$CONFIG_ID" \
    --output_file="$TASK_FILE" \
    --log_dir="$RUN_LOGS" \
    --extra_args="--verbose --output-dir /mnt/ceph/users/tparshakova/step-back/output"

echo "Submitting to Slurm..."

export TASK_FILE

sbatch \
    --partition=gpu \
    --ntasks=18 \
    --cpus-per-task=4 \
    --gpus-per-task=1 \
    --mem-per-gpu=8G \
    --time=11:00:00 \
    --job-name="$CONFIG_ID" \
    --output=slurm_logs/slurm_job_%j.out \
    --error=slurm_logs/slurm_job_%j.err \
    slurm_scripts/sbatch.sh

echo "Submitted. Check slurm_logs/ and ${RUN_LOGS}/ for output."
