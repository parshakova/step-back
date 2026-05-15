"""
Run a single experiment from a config file, selected by job index.

Designed for parallel execution via disBatch on Slurm: each task runs one
experiment configuration from the cartesian product defined in the config.

Usage:
    python run_hydra.py -i <config_id> -j <job_index> [options]

Example:
    python run_hydra.py -i cifar10_resnet20_compare -j 0
    python run_hydra.py -i cifar10_resnet20_compare -j 3 --verbose
    python run_hydra.py -i cifar10_resnet20_compare -j 0 --no-wandb
"""
import argparse
import os
import torch
from typing import Union

from stepback.base import Base
from stepback.log import Container
from stepback.config import ConfigManager
from stepback.defaults import DEFAULTS

parser = argparse.ArgumentParser(description='Run a single experiment by index (for Slurm/disBatch).')
parser.add_argument('-i', '--id', nargs='?', type=str, required=True, help="The id of the config (its file name).")
parser.add_argument('-j', '--job-index', type=int, required=True, help="Index of the experiment in the config list.")
parser.add_argument('-cdir', '--config-dir', nargs='?', type=str, default=DEFAULTS.config_dir, help="The config directory.")
parser.add_argument('-odir', '--output-dir', nargs='?', type=str, default=DEFAULTS.output_dir, help="The output directory.")
parser.add_argument('-ddir', '--data-dir', nargs='?', type=str, default=DEFAULTS.data_dir, help="The data directory.")
parser.add_argument('--device', nargs='?', type=str, default=DEFAULTS.device, help="Device to run on.")
parser.add_argument('-nw', '--num-workers', nargs='?', type=int, default=DEFAULTS.num_workers, help="Number of workers for DataLoader.")
parser.add_argument('--data-parallel', nargs='+', default=DEFAULTS.data_parallel, help='Device list for DataParallel in Pytorch.')
parser.add_argument('-logk', '--log-every-k-steps', nargs='?', type=int, default=DEFAULTS.log_every_k_steps, help="Stepwise logging.")
parser.add_argument('--verbose', action="store_true", help="Verbose mode.")
parser.add_argument('--force-deterministic', action="store_true", help="Use deterministic mode in Pytorch.")
parser.add_argument('--no-wandb', action="store_true", help="Disable wandb logging.")

WANDB_PROJECT = "spectral_clip"


def init_wandb(exp_id, job_index, config):
    """Initialize wandb run with Slurm-aware naming and offline directory on ceph."""
    import wandb

    job_id = os.environ.get("SLURM_JOB_ID", "nojob")
    proc_id = os.environ.get("SLURM_PROCID", "0")
    slurm_job_name = os.environ.get("SLURM_JOB_NAME", exp_id)

    # Build run name from optimizer config
    opt = config.get('opt', {})
    opt_name = opt.get('name', 'unknown')
    lr = opt.get('lr', '')
    run_name = f"{opt_name}_lr{lr}_j{job_index}"

    # Set wandb directory to ceph for offline storage
    base_wandb_dir = os.environ.get(
        "WANDB_DIR",
        "/mnt/ceph/users/tparshakova/wandb_offline"
    )
    wandb_root = os.path.join(base_wandb_dir, str(slurm_job_name), str(job_id))
    os.makedirs(wandb_root, exist_ok=True)

    tags = [exp_id, opt_name, f"j{job_id}-p{proc_id}"]

    try:
        wandb_run = wandb.init(
            project=WANDB_PROJECT,
            name=run_name,
            config=config,
            dir=wandb_root,
            tags=tags,
            reinit=True,
        )
    except Exception as e:
        print(f"wandb online init failed ({e}), falling back to offline mode.")
        os.environ["WANDB_MODE"] = "offline"
        wandb_run = wandb.init(
            project=WANDB_PROJECT,
            name=run_name,
            config=config,
            dir=wandb_root,
            tags=tags,
            reinit=True,
            mode="offline",
        )

    return wandb_run


def run_single(
        exp_id: str,
        job_index: int,
        config_dir: str = DEFAULTS.config_dir,
        output_dir: str = DEFAULTS.output_dir,
        data_dir: str = DEFAULTS.data_dir,
        device: str = DEFAULTS.device,
        num_workers: int = DEFAULTS.num_workers,
        data_parallel: Union[list, None] = DEFAULTS.data_parallel,
        log_every_k_steps: Union[int, None] = DEFAULTS.log_every_k_steps,
        verbose: bool = DEFAULTS.verbose,
        force_deterministic: bool = DEFAULTS.force_deterministic,
        no_wandb: bool = False
    ):
    """Run a single experiment configuration selected by job_index.

    Parameters
    ----------
    exp_id : str
        The experiment ID, equal to the name of the config file.
    job_index : int
        Index into the cartesian product of configs to run.
    no_wandb : bool
        If True, disable wandb logging.
    """
    # Load config and expand into list
    Conf = ConfigManager(exp_id=exp_id, config_dir=config_dir)
    exp_list = Conf.create_config_list()

    if job_index < 0 or job_index >= len(exp_list):
        raise ValueError(f"job_index {job_index} out of range [0, {len(exp_list)})")

    config = exp_list[job_index]
    print(f"Running experiment {exp_id}, job {job_index}/{len(exp_list)}: {config.get('opt', {})}")

    if force_deterministic:
        torch.use_deterministic_algorithms(True)
        print("Using Pytorch deterministic mode.")

    # Initialize wandb
    wandb_run = None
    if not no_wandb:
        wandb_run = init_wandb(exp_id, job_index, config)

    # Output file per job to avoid write conflicts
    output_name = f"{exp_id}-{job_index:03d}"

    try:
        B = Base(
            name=output_name,
            config=config,
            device=device,
            data_dir=data_dir,
            num_workers=num_workers,
            data_parallel=data_parallel,
            log_every_k_steps=log_every_k_steps,
            verbose=verbose,
            wandb_run=wandb_run
        )

        B.setup()
        B.run()

        C = Container(name=output_name, output_dir=output_dir, as_json=True)
        C.append(B.results).store()

        print(f"Job {job_index} completed. Results saved to {output_dir}/{output_name}.json")
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == '__main__':
    args = parser.parse_args()

    run_single(
        args.id,
        job_index=args.job_index,
        config_dir=args.config_dir,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        device=args.device,
        num_workers=args.num_workers,
        data_parallel=args.data_parallel,
        log_every_k_steps=args.log_every_k_steps,
        verbose=args.verbose,
        force_deterministic=args.force_deterministic,
        no_wandb=args.no_wandb
    )