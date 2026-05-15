#!/bin/bash
#SBATCH -p gpu
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/slurm_job_%j.out
#SBATCH --error=slurm_logs/slurm_job_%j.err

echo "Starting sbatch.sh on node $(hostname)..."
echo "Job ID: $SLURM_JOB_ID"
echo "TASK_FILE: $TASK_FILE"

mkdir -p slurm_logs

module load disBatch
module load python

# Activate virtual environment if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

echo "Starting disBatch with task file $TASK_FILE..."
disBatch $TASK_FILE --prefix "disbatch_logs"
echo "disBatch completed for Job ID $SLURM_JOB_ID"

wait