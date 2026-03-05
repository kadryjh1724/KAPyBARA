"""cli/monitor.py — 'kapybara monitor' handler.

ASCII monitoring board. Replaces the .chk/.jobID file scan with a single
StateDB query per cell.

Cell colours:
  black bg  = not started (pending/submitted)
  green bg  = running   (shows number of incomplete replicas)
  blue bg   = completed
  red bg    = failed
"""

import os
import sys
import time
import signal
import subprocess

from kapybara.config.loader import load_config
from kapybara.config.paths import PathManager
from kapybara.state.db import StateDB

MIN_X_WIDTH = 50


def _signal_handler(sig, frame):
    """Handle SIGINT by printing a termination message and exiting cleanly."""
    print("\nMonitoring terminated.")
    sys.exit(0)


def _not_started():
    """Print a black-background 2-character cell (not started)."""
    print("\x1b[6;30;40m" + "  " + "\x1b[0m", end="")


def _running(n_remaining: int):
    """Print a green-background 2-digit cell showing remaining replica count.

    Args:
        n_remaining: Number of replicas not yet completed.
    """
    print("\x1b[6;30;42m" + f"{n_remaining:02d}" + "\x1b[0m", end="")


def _completed():
    """Print a blue-background 2-character cell (completed)."""
    print("\x1b[0;30;44m" + "  " + "\x1b[0m", end="")


def _failed():
    """Print a red-background 2-character cell (failed)."""
    print("\x1b[0;30;41m" + "  " + "\x1b[0m", end="")


def _get_pid(config_path: str) -> str:
    """Find the PID of the running ``kapybara run`` process for this config.

    Uses ``ps aux | grep`` to locate a process matching the config file name,
    excluding monitor and queue processes. Returns ``"N/A"`` if none found.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        PID string of the first matching process, or ``"N/A"``.
    """
    config_name = os.path.basename(config_path)
    cmd = (
        f"ps aux | grep 'kapybara.*{config_name}'"
        f" | grep -v grep"
        f" | grep -v 'kapybara monitor'"
        f" | grep -v 'kapybara queue'"
        f" | awk '{{print $2}}'"
    )
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    pids = [p for p in result.stdout.strip().split("\n") if p]
    if not pids:
        return "N/A"
    return pids[0]


def _cell_status(state_db: StateDB, T: str, field_value: str,
                 n_replica: int) -> None:
    """Print the coloured status cell for one (T, field_value) point."""
    status = state_db.get_tps_job_status(T, field_value)

    if status == "completed":
        _completed()
    elif status in ("submitted", "running"):
        progress = state_db.get_tps_replica_progress(T, field_value)
        n_done = sum(1 for row in progress if row["status"] == "completed")
        _running(n_replica - n_done)
    elif status == "failed":
        _failed()
    else:
        _not_started()


def _print_board(config, state_db: StateDB, args) -> None:
    """Render one frame of the ASCII monitoring board to stdout.

    Determines which axis (T or field) to place on the x-axis based on
    which has more values, then prints a coloured 2-char cell for each
    (T, field) job using the current StateDB state.

    Args:
        config: Frozen :class:`~kapybara.config.schema.SimulationConfig`.
        state_db: :class:`~kapybara.state.db.StateDB` instance.
        args: Parsed :class:`argparse.Namespace` (used for ``args.watch`` and
            ``args.config``).
    """
    T_vals = config.T
    field_vals = config.g if config.runtype == "g" else config.s

    n_T     = len(T_vals)
    n_field = len(field_vals)

    if n_T >= n_field:
        x, y     = T_vals, field_vals[::-1]
        xlabel, ylabel = "T", config.runtype
    else:
        x, y     = field_vals, T_vals[::-1]
        xlabel, ylabel = config.runtype, "T"

    dim_x = max(MIN_X_WIDTH, 2 * len(x))
    dim_y = len(y)

    print(f'╔{"═" * dim_x}╗')
    print(f"║" + "KAPYBARA MONITORING BOARD".center(dim_x) + "║")
    print(f'╠{"═" * 22}╦{"═" * (dim_x - 23)}╣')
    print(f"║" + " JOB NAME".ljust(22) + "║ "
          + f"{config.job_name}".ljust(dim_x - 24) + "║")
    print(f"║" + " PROCESS PID".ljust(22) + "║ "
          + f"{_get_pid(args.config)}".ljust(dim_x - 24) + "║")
    print(f"║" + f" (T, {config.runtype}) DIMENSION".ljust(22) + "║ "
          + f"({n_T}, {n_field})".ljust(dim_x - 24) + "║")
    print(f"║" + " NUM. OF REPLICAS".ljust(22) + "║ "
          + f"{config.n_replica}".ljust(dim_x - 24) + "║")
    print(f'╠{"═" * 22}╩{"═" * (dim_x - 23)}╣')
    print(
        f"║" + " " * ((dim_x - 48) // 2)
        + "\x1b[6;30;40m  \x1b[0m NOT STARTED "
        + "\x1b[6;30;42m  \x1b[0m RUNNING "
        + "\x1b[0;30;44m  \x1b[0m COMPLETED "
        + "\x1b[0;30;41m  \x1b[0m FAILED"
        + " " * ((dim_x - 48) // 2) + "║"
    )
    if args.watch:
        print(f"║"
              + f"UPDATE EVERY {args.watch} SECONDS, Ctrl+C TO EXIT".center(dim_x)
              + "║")
    print(f'╠{"═" * dim_x}╣')

    for i, y_val in enumerate(y):
        print(f"║", end="")
        print("\x1b[0;30;46m" + " " * ((dim_x - 2 * len(x)) // 2) + "\x1b[0m",
              end="")

        for x_val in x:
            if xlabel == "T":
                t_point, f_point = x_val, y_val
            else:
                f_point, t_point = x_val, y_val
            _cell_status(state_db, t_point, f_point, config.n_replica)

        print("\x1b[0;30;46m" + " " * ((dim_x - 2 * len(x)) // 2) + "\x1b[0m",
              end="")

        if i == dim_y // 2:
            print(f"║ {ylabel}")
        else:
            print(f"║")

    print(f'╚{"═" * dim_x}╝')
    print(f"{xlabel}".center(dim_x + 2))


def monitor(args) -> None:
    """Handle the ``kapybara monitor`` sub-command.

    Renders a colour-coded ASCII board showing TPS job progress for every
    (T, field) combination. In watch mode (``args.watch`` is set), clears
    the terminal and re-renders every ``args.watch`` seconds until Ctrl+C.

    Args:
        args: Parsed :class:`argparse.Namespace` with attributes:
            ``config`` (str), ``watch`` (int or None).
    """
    config   = load_config(args.config, quiet=True)
    paths    = PathManager(config)
    state_db = StateDB(paths.db)

    if config.runtype not in ("g", "s"):
        print(f"kapybara monitor: not implemented for runtype={config.runtype}.")
        return

    while True:
        if args.watch:
            signal.signal(signal.SIGINT, _signal_handler)
            os.system("clear")

        _print_board(config, state_db, args)

        if args.watch is None:
            break
        time.sleep(args.watch)
