"""TPS run worker entry point.

Launched by SLURM sbatch --wrap "python -m kapybara.commands.run".
Runs specified replicas for one (T, field_value) point in parallel via
multiprocessing. The run_type is determined from the config (g, s, or sg).

Single-writer architecture: one DBWriter process owns all SQLite writes.
Worker processes send write requests via a multiprocessing.Queue.
"""

import argparse
import multiprocessing as mp

from kapybara.config.loader import load_config
from kapybara.config.paths import PathManager
from kapybara.state.db import StateDB
from kapybara.state.writer import DBWriter
from kapybara.core.thermostat import create_thermostat
from kapybara.sampling.moves import TPSMoves
from kapybara.sampling.runners import RunnerTg, RunnerTs, RunnerSg

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config",      required=True, type=str)
parser.add_argument("-q", "--quiet",       action="store_true", default=False)
parser.add_argument("-t", "--temperature", required=True, type=str)
parser.add_argument("-f", "--field",       required=True, type=str,
                    help="Field value (g or s depending on run type).")
parser.add_argument("-r", "--replicas",    required=True, type=str,
                    help="Comma-separated replica indices to run (e.g. '0,1,3').")
args = parser.parse_args()


def _create_runner(config, paths, state_db, moves):
    """Instantiate the appropriate Runner for the configured run type.

    Args:
        config: Frozen :class:`~kapybara.config.schema.SimulationConfig`.
        paths: :class:`~kapybara.config.paths.PathManager` instance.
        state_db: :class:`~kapybara.state.db.StateDB` instance (queue-mode).
        moves: :class:`~kapybara.sampling.moves.TPSMoves` instance.

    Returns:
        A concrete runner instance (RunnerTg, RunnerTs, or RunnerSg).

    Raises:
        ValueError: If ``config.runtype`` is not ``"g"``, ``"s"``, or ``"sg"``.
    """
    match config.runtype:
        case "g":
            return RunnerTg(config, paths, state_db, moves)
        case "s":
            return RunnerTs(config, paths, state_db, moves)
        case "sg":
            return RunnerSg(config, paths, state_db, moves)
        case _:
            raise ValueError(f"Unknown runtype: {config.runtype}")


def _run_replica(replica_index: int, write_queue) -> None:
    """Worker function: run the TPS workflow for a single replica.

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

    if state_db.is_tps_replica_completed(args.temperature, args.field, replica_index):
        return

    thermostat = create_thermostat(config)
    moves      = TPSMoves(config, thermostat)
    runner     = _create_runner(config, paths, state_db, moves)
    runner.run(args.temperature, args.field, replica_index)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    config = load_config(args.config, quiet=True)
    paths  = PathManager(config)

    # Ensure schema exists before spawning any subprocesses
    StateDB(paths.db)

    # Direct-mode StateDB for main-process job-level status update
    state_db = StateDB(paths.db)
    state_db.update_tps_job_status(args.temperature, args.field, "running")

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
