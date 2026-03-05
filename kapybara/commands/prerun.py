"""Prerun worker entry point.

Launched by SLURM sbatch --wrap "python -m kapybara.commands.prerun".
Runs specified replicas for one temperature in parallel via multiprocessing.

Single-writer architecture: one DBWriter process owns all SQLite writes.
Worker processes send write requests via a multiprocessing.Queue.
"""

import argparse
import multiprocessing as mp

from kapybara.config.loader import load_config
from kapybara.config.paths import PathManager
from kapybara.state.db import StateDB
from kapybara.state.writer import DBWriter
from kapybara.prepare.prepare import Prepare

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config",      required=True,  type=str)
parser.add_argument("-q", "--quiet",       action="store_true", default=False)
parser.add_argument("-t", "--temperature", required=True,  type=str)
parser.add_argument("-r", "--replicas",    required=True,  type=str,
                    help="Comma-separated replica indices to run (e.g. '0,1,3').")
args = parser.parse_args()


def _run_replica(replica_index: int, write_queue) -> None:
    """Worker function: run the prerun workflow for a single replica.

    Intended to be executed inside a :class:`multiprocessing.Process`. Skips
    the replica immediately if StateDB already shows it as completed (safe
    restart behaviour).

    Args:
        replica_index: Zero-based index of the replica to run.
        write_queue: Multiprocessing queue shared with the
            :class:`~kapybara.state.writer.DBWriter` subprocess.
    """
    config   = load_config(args.config, quiet=True)
    paths    = PathManager(config)
    state_db = StateDB(paths.db, write_queue=write_queue)

    if state_db.is_prerun_replica_completed(args.temperature, replica_index):
        return

    prepare = Prepare(config, paths, state_db)
    prepare.prerun(args.temperature, replica_index)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    config = load_config(args.config, quiet=True)
    paths  = PathManager(config)

    # Ensure schema exists before spawning any subprocesses
    StateDB(paths.db)

    # Direct-mode StateDB for main-process job-level status updates
    state_db = StateDB(paths.db)
    state_db.update_prerun_job_status(args.temperature, "running")

    replica_indices = [int(r) for r in args.replicas.split(",")]

    writer = DBWriter(paths.db)
    writer.start()

    processes = []
    for i in replica_indices:
        p = mp.Process(target=_run_replica, args=(i, writer.queue))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    writer.stop()

    # Mark job completed only when all n_replica replicas are done
    if state_db.is_prerun_completed(args.temperature, config.n_replica):
        state_db.update_prerun_job_status(args.temperature, "completed")
