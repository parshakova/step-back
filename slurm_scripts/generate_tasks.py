"""
Generate a disBatch task file from a step-back config.

Expands the config's cartesian product and writes one task line per experiment,
each calling run_hydra.py with the corresponding job index.

Usage:
    python slurm_scripts/generate_tasks.py \
        --config_id cifar10_resnet20_compare \
        --output_file tasks.txt \
        [--config_dir configs/] \
        [--log_dir logs/] \
        [--extra_args "--verbose"]
"""
import json
import os
import sys
import argparse

# Add project root to path so we can import stepback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stepback.config import ConfigManager


def main():
    parser = argparse.ArgumentParser(description='Generate disBatch task file from step-back config.')
    parser.add_argument('--config_id', required=True, help='Config file name (without .json)')
    parser.add_argument('--output_file', required=True, help='Path to write the task file')
    parser.add_argument('--config_dir', default='configs/', help='Config directory')
    parser.add_argument('--log_dir', default=None, help='Directory for per-task log files')
    parser.add_argument('--extra_args', default='', help='Extra arguments to pass to run_hydra.py')
    args = parser.parse_args()

    # Expand config to get total number of experiments
    Conf = ConfigManager(exp_id=args.config_id, config_dir=args.config_dir)
    exp_list = Conf.create_config_list()
    n_tasks = len(exp_list)
    num_digits = len(str(n_tasks))

    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)

    with open(args.output_file, 'w') as f:
        for i in range(n_tasks):
            cmd = f"python run_hydra.py -i {args.config_id} -j {i}"
            if args.extra_args:
                cmd += f" {args.extra_args}"

            if args.log_dir:
                padded = str(i).zfill(num_digits)
                cmd += f" > {args.log_dir}/log_{padded}.out 2> {args.log_dir}/log_{padded}.err"

            f.write(cmd + "\n")

    print(f"Generated {n_tasks} tasks in {args.output_file}")


if __name__ == '__main__':
    main()