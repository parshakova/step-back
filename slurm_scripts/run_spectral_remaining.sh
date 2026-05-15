#!/bin/bash
#
# Run missing spectral-clip (jobs 5-9) and spectral-inv (jobs 20-24)
# experiments from cifar10_resnet20_compare.json.
#
# Usage:
#   ./slurm_scripts/run_spectral_remaining.sh

set -e

CONFIG_ID="cifar10_resnet20_compare"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs/${CONFIG_ID}"
RUN_LOGS="${LOG_DIR}/run_logs"
TASK_FILE="${LOG_DIR}/tasks_spectral_remaining"

mkdir -p "$LOG_DIR" "$RUN_LOGS" "${ROOT_DIR}/slurm_logs" "${ROOT_DIR}/disbatch_logs" "${ROOT_DIR}/output"

cd "$ROOT_DIR"

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

echo "=== ${CONFIG_ID}: spectral-clip (jobs 5-9) + spectral-inv (jobs 20-24) ==="

# Generate task file for the 10 missing jobs
cat > "$TASK_FILE" <<'EOF'
python run_hydra.py -i cifar10_resnet20_compare -j 5 --verbose > logs/cifar10_resnet20_compare/run_logs/log_05.out 2> logs/cifar10_resnet20_compare/run_logs/log_05.err
python run_hydra.py -i cifar10_resnet20_compare -j 6 --verbose > logs/cifar10_resnet20_compare/run_logs/log_06.out 2> logs/cifar10_resnet20_compare/run_logs/log_06.err
python run_hydra.py -i cifar10_resnet20_compare -j 7 --verbose > logs/cifar10_resnet20_compare/run_logs/log_07.out 2> logs/cifar10_resnet20_compare/run_logs/log_07.err
python run_hydra.py -i cifar10_resnet20_compare -j 8 --verbose > logs/cifar10_resnet20_compare/run_logs/log_08.out 2> logs/cifar10_resnet20_compare/run_logs/log_08.err
python run_hydra.py -i cifar10_resnet20_compare -j 9 --verbose > logs/cifar10_resnet20_compare/run_logs/log_09.out 2> logs/cifar10_resnet20_compare/run_logs/log_09.err
python run_hydra.py -i cifar10_resnet20_compare -j 20 --verbose > logs/cifar10_resnet20_compare/run_logs/log_20.out 2> logs/cifar10_resnet20_compare/run_logs/log_20.err
python run_hydra.py -i cifar10_resnet20_compare -j 21 --verbose > logs/cifar10_resnet20_compare/run_logs/log_21.out 2> logs/cifar10_resnet20_compare/run_logs/log_21.err
python run_hydra.py -i cifar10_resnet20_compare -j 22 --verbose > logs/cifar10_resnet20_compare/run_logs/log_22.out 2> logs/cifar10_resnet20_compare/run_logs/log_22.err
python run_hydra.py -i cifar10_resnet20_compare -j 23 --verbose > logs/cifar10_resnet20_compare/run_logs/log_23.out 2> logs/cifar10_resnet20_compare/run_logs/log_23.err
python run_hydra.py -i cifar10_resnet20_compare -j 24 --verbose > logs/cifar10_resnet20_compare/run_logs/log_24.out 2> logs/cifar10_resnet20_compare/run_logs/log_24.err
EOF

echo "Task file: $TASK_FILE"
echo ""
echo "Submitting to Slurm..."

export TASK_FILE

sbatch \
    --partition=gpu \
    --ntasks=10 \
    --cpus-per-task=4 \
    --gpus-per-task=1 \
    --mem-per-gpu=8G \
    --time=02:00:00 \
    --job-name="${CONFIG_ID}_spectral" \
    --output=slurm_logs/slurm_job_%j.out \
    --error=slurm_logs/slurm_job_%j.err \
    slurm_scripts/sbatch.sh

echo "Submitted. Check slurm_logs/ and ${RUN_LOGS}/ for output."
