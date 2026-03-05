"""cli/run.py — 'kapybara run' handler.

Loads configuration, constructs the dependency DAG and Scheduler,
then enters the main scheduling loop. run_type is determined from the
config file (g / s / sg) — no separate 'run_Tg' command needed.
"""

from kapybara.config.loader import load_config
from kapybara.config.paths import PathManager
from kapybara.state.db import StateDB
from kapybara.orchestrate.dag import DependencyDAG
from kapybara.orchestrate.scheduler import Scheduler


def run(args) -> None:
    """Handle the ``kapybara run`` sub-command.

    Loads configuration, constructs the field-dependency DAG and Scheduler,
    initialises StateDB job records, and enters the main scheduling loop.
    The loop runs until all TPS jobs for all (T, field) points complete.

    Args:
        args: Parsed :class:`argparse.Namespace` with attributes:
            ``config`` (str), ``quiet`` (bool).
    """
    import os

    config      = load_config(args.config, quiet=args.quiet)
    paths       = PathManager(config)
    paths.ensure_directories()
    state_db    = StateDB(paths.db)
    dag         = DependencyDAG(config)
    config_path = os.path.realpath(args.config)

    scheduler = Scheduler(
        config      = config,
        paths       = paths,
        state_db    = state_db,
        dag         = dag,
        config_path = config_path,
        quiet       = args.quiet,
    )
    scheduler.initialize()
    scheduler.run()
