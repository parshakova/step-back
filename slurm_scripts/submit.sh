#!/bin/bash
#
# Submit a step-back config as parallel disBatch tasks on Slurm.
#
# Usage:
#   ./slurm_scripts/submit.sh <config_id> <num_gpus> [time_days] [extra_args]
#
# Examples:
#   ./slurm_scripts/submit.sh cifar10_resnet20_compare 15
#   ./slurm_scripts/submit.sh cifar10_resnet20_compare 15 2
#   ./slurm_scripts/submit.sh cifar10_resnet20_compare 15 2 "--verbose"

set -e

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <config_id> <num_gpus> [time_days] [extra_args]"
    echo ""
    echo "  config_id   : Name of the config file (without .json)"
    echo "  num_gpus    : Number of GPUs (= max parallel tasks)"
    echo "  time_days   : Wall time in days (default: 1)"
    echo "  extra_args  : Extra arguments for run_hydra.py (e.g. \"--verbose\")"
    exit 1
fi

CONFIG_ID=$1
NUM_GPUS=$2
TIME_DAYS=${3:-1}
EXTRA_ARGS=${4:-""}

# Choose partition via env var (default: gpu)
PARTITION=${PARTITION:-gpu}
GPU_CONSTRAINT=${GPU_CONSTRAINT:-""}

# Set up directories
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs/${CONFIG_ID}"
TASK_FILE="${LOG_DIR}/tasks"
RUN_LOGS="${LOG_DIR}/run_logs"

mkdir -p "$LOG_DIR" "$RUN_LOGS" "${ROOT_DIR}/slurm_logs" "${ROOT_DIR}/disbatch_logs"

echo "=== step-back submit ==="
echo "Config:     $CONFIG_ID"
echo "GPUs:       $NUM_GPUS"
echo "Time:       ${TIME_DAYS} day(s)"
echo "Partition:  $PARTITION"
echo "Extra args: $EXTRA_ARGS"

# Generate task file
echo "Generating task file..."
cd "$ROOT_DIR"
python slurm_scripts/generate_tasks.py \
    --config_id="$CONFIG_ID" \
    --output_file="$TASK_FILE" \
    --log_dir="$RUN_LOGS" \
    --extra_args="$EXTRA_ARGS"

if [ $? -ne 0 ]; then
    echo "Error: Failed to generate task file."
    exit 1
fi

# Export for sbatch.sh
export TASK_FILE
export LOG_DIR

# Submit
echo "Submitting to Slurm..."
sbatch \
    --partition="$PARTITION" \
    --ntasks="$NUM_GPUS" \
    --time="${TIME_DAYS}-00:00:00" \
    ${GPU_CONSTRAINT:+--constraint="$GPU_CONSTRAINT"} \
    --job-name="$CONFIG_ID" \
    slurm_scripts/sbatch.sh

echo "Job submitted. Check slurm_logs/ for output."
