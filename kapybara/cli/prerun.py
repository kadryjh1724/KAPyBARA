"""cli/prerun.py — 'kapybara prerun' handler.

Submits one SLURM prerun job per temperature that has incomplete replicas.
On re-run, only pending/failed replicas are targeted, with CPUs allocated
exactly to match the number of replicas that still need to run.
"""

import os

from kapybara.config.loader import load_config
from kapybara.config.paths import PathManager
from kapybara.state.db import StateDB
from kapybara.orchestrate.slurm import (
    construct_sbatch_command,
    submit_job,
    set_mpi_environment,
)


def prerun(args) -> None:
    """Handle the ``kapybara prerun`` sub-command.

    For each temperature, determines which replicas are not yet completed
    and submits a single SLURM job with one CPU per pending replica. Jobs
    are skipped entirely for temperatures where all replicas are already done.

    Args:
        args: Parsed :class:`argparse.Namespace` with attributes:
            ``config`` (str), ``quiet`` (bool).
    """
    set_mpi_environment()

    config = load_config(args.config, quiet=args.quiet)
    paths  = PathManager(config)
    paths.ensure_directories()
    state_db = StateDB(paths.db)

    config_path = os.path.realpath(args.config)

    for T in config.T:
        pending = state_db.get_pending_prerun_replicas(T, config.n_replica)

        if not pending:
            print(f"kapybara prerun: T={T} — all {config.n_replica} replicas complete, skipping.")
            continue

        n_pending = len(pending)
        if n_pending < config.n_replica:
            print(f"kapybara prerun: T={T} — {n_pending}/{config.n_replica} replicas pending, resubmitting.")

        state_db.register_prerun_job(T)

        replicas_str = ",".join(str(r) for r in pending)
        worker_cmd = (
            f"srun --mpi=pmi2 python -m kapybara.commands.prerun"
            f" -c {config_path} -t {T} -r {replicas_str}"
        )
        if args.quiet:
            worker_cmd += " -q"

        cmd = construct_sbatch_command(
            job_name    = f"{config.job_name}-prerun-{T}",
            partition   = config.partition,
            n_tasks     = n_pending,
            stdout_path = os.path.join(paths.step1_stdout, f"prerun-{T}.out"),
            stderr_path = os.path.join(paths.step1_stderr, f"prerun-{T}.err"),
            command     = worker_cmd,
            exclude     = config.exclude,
        )
        job_id = submit_job(cmd)
        state_db.submit_prerun_job(T, job_id)
        print(f"kapybara prerun: T={T} — submitted job {job_id} ({n_pending} CPUs).")
