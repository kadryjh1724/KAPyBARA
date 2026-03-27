"""cli/run.py — 'kapybara run' handler.

Loads configuration, constructs the dependency DAG and Scheduler,
then enters the main scheduling loop. run_type is determined from the
config file (g / s / sg) — no separate 'run_Tg' command needed.

With ``--bg``, the scheduler is detached into the background and the
parent process exits immediately after printing the PID and log path.
Without ``--bg``, the scheduler runs in the foreground; SIGTERM and
SIGINT are caught for clean shutdown.
"""

import os
import sys
import signal
import subprocess

from kapybara.config.loader import load_config
from kapybara.config.paths import PathManager
from kapybara.state.db import StateDB
from kapybara.orchestrate.dag import DependencyDAG
from kapybara.orchestrate.scheduler import Scheduler


class ShutdownRequested(Exception):
    """Raised by the SIGTERM / SIGINT handler to break the scheduler loop cleanly.

    Using an exception instead of a flag ensures that ``time.sleep()`` inside
    the scheduler loop is interrupted immediately on signal delivery.
    """


def _handle_term(signum, frame):
    """Signal handler for SIGTERM and SIGINT.

    Raises :class:`ShutdownRequested` so the foreground scheduler loop exits
    cleanly on ``Ctrl+C`` or ``kapybara stop``.

    Args:
        signum: Signal number (unused beyond triggering the raise).
        frame: Current stack frame (unused).
    """
    raise ShutdownRequested()


def _run_background(args, paths) -> None:
    """Re-launch the scheduler as a detached background process.

    Spawns a new process running ``python -m kapybara.cli run`` without
    ``--bg``, redirecting both stdout and stderr to *log_path* so all
    scheduler print output (job submission messages, completion notices,
    errors) is captured there instead of the terminal. The child process
    starts a new session (``start_new_session=True``) so it survives
    terminal closure.

    Args:
        args: Parsed :class:`argparse.Namespace` with attributes:
            ``config`` (str), ``log`` (str or None — path for stdout/stderr
            capture), ``quiet`` (bool).
        paths: :class:`~kapybara.config.paths.PathManager` for default log dir.
    """
    config_path = os.path.realpath(args.config)
    log_path = args.log or os.path.join(paths.base, "kapybara.log")

    cmd = [sys.executable, "-m", "kapybara", "run", "-c", config_path, "-q"]
    log_fd = open(log_path, "a")
    proc = subprocess.Popen(
        cmd,
        stdout=log_fd,
        stderr=log_fd,
        start_new_session=True,
    )
    log_fd.close()

    print(f"kapybara: backgrounded (PID {proc.pid})")
    print(f"kapybara: log → {log_path}")


def _run_foreground(args, config, paths) -> None:
    """Run the scheduler in the foreground with signal handling.

    Installs SIGTERM and SIGINT handlers that raise
    :class:`ShutdownRequested`, so both ``kapybara stop`` and ``Ctrl+C``
    produce a clean exit message instead of a traceback.

    Args:
        args: Parsed :class:`argparse.Namespace` with attributes:
            ``config`` (str), ``quiet`` (bool).
        config: Frozen :class:`~kapybara.config.schema.SimulationConfig`.
        paths: :class:`~kapybara.config.paths.PathManager` instance.
    """
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

    signal.signal(signal.SIGTERM, _handle_term)
    signal.signal(signal.SIGINT,  _handle_term)

    try:
        scheduler.run()
    except ShutdownRequested:
        print("\nkapybara: scheduler stopped by signal.")


def run(args) -> None:
    """Handle the ``kapybara run`` sub-command.

    Loads configuration and dispatches to either the background launcher
    (``--bg``) or the foreground scheduling loop.

    Args:
        args: Parsed :class:`argparse.Namespace` with attributes:
            ``config`` (str), ``quiet`` (bool), ``bg`` (bool),
            ``log`` (str or None).
    """
    config = load_config(args.config, quiet=args.quiet)
    paths  = PathManager(config)

    if args.bg:
        _run_background(args, paths)
    else:
        _run_foreground(args, config, paths)
