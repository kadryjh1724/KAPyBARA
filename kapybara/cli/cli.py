#!/usr/bin/env python3
"""kapybara — main CLI entry point.

Subcommands:
  prerun   Submit prerun SLURM jobs (one per temperature).
  run      Enter the TPS scheduling loop (run_type from config).
  monitor  ASCII monitoring board backed by StateDB.
  queue    SLURM queue viewer with per-job progress bar.
  analysis Analysis sub-commands (acceptance / time / data).
"""

import argparse

from kapybara.cli.prerun  import prerun
from kapybara.cli.run     import run
from kapybara.cli.monitor import monitor
from kapybara.cli.queue   import queue


def main():
    """Entry point for the ``kapybara`` command-line tool.

    Parses the subcommand and delegates to the appropriate handler function
    (prerun, run, monitor, queue, or analysis). Analysis sub-commands are
    registered but not yet implemented in this version.
    """
    parser = argparse.ArgumentParser(
        prog        = "kapybara",
        description = (
            "KAPyBARA: Kob-Andersen model with Python-BAsed tRAjectory sampling, "
            "powered by LAMMPS."
        ),
        epilog = "Developed by: Jiho Son (kadryjh1724@snu.ac.kr)",
    )
    subparsers = parser.add_subparsers(
        title    = "Available subcommands",
        dest     = "command",
        required = True,
    )

    # ── prerun ────────────────────────────────────────────────────────────
    p_prerun = subparsers.add_parser(
        "prerun", help="Submit prerun SLURM jobs."
    )
    p_prerun.add_argument("-c", "--config", required=True, type=str,
                          help="Path to config YAML file.")
    p_prerun.add_argument("-q", "--quiet", action="store_true", default=False,
                          help="Suppress output messages.")
    p_prerun.set_defaults(func=lambda args: prerun(args))

    # ── run ───────────────────────────────────────────────────────────────
    p_run = subparsers.add_parser(
        "run", help="Start TPS scheduling loop (run_type from config)."
    )
    p_run.add_argument("-c", "--config", required=True, type=str,
                       help="Path to config YAML file.")
    p_run.add_argument("-q", "--quiet", action="store_true", default=False,
                       help="Suppress output messages.")
    p_run.set_defaults(func=lambda args: run(args))

    # ── monitor ───────────────────────────────────────────────────────────
    p_monitor = subparsers.add_parser(
        "monitor", help="ASCII monitoring board."
    )
    p_monitor.add_argument("-c", "--config", required=True, type=str,
                           help="Path to config YAML file.")
    p_monitor.add_argument("-w", "--watch", type=int, required=False,
                           help="Refresh interval in seconds.")
    p_monitor.set_defaults(func=lambda args: monitor(args))

    # ── queue ─────────────────────────────────────────────────────────────
    p_queue = subparsers.add_parser(
        "queue", help="SLURM queue viewer with progress bar."
    )
    p_queue.add_argument("-c", "--config", required=True, type=str,
                         help="Path to config YAML file.")
    p_queue.add_argument("-w", "--watch", type=int, required=False,
                         help="Refresh interval in seconds.")
    p_queue.add_argument("-n", "--number", type=int, required=False,
                         default=20,
                         help="Show top N entries from squeue.")
    p_queue.set_defaults(func=lambda args: queue(args))

    # ── analysis ──────────────────────────────────────────────────────────
    p_analysis = subparsers.add_parser(
        "analysis", help="Analysis tools."
    )
    p_analysis.add_argument("-c", "--config", required=True, type=str,
                            help="Path to config YAML file.")
    analysis_sub = p_analysis.add_subparsers(
        title="Analysis subcommands", dest="analysis_command", required=True
    )

    p_acc = analysis_sub.add_parser("acceptance",
                                    help="Compute acceptance statistics.")
    p_acc.add_argument("-m", "--colormap", type=str, default="inferno")
    p_acc.add_argument("--title",       action="store_true")
    p_acc.add_argument("--transparent", action="store_true")
    p_acc.set_defaults(func=lambda args: _not_implemented("analysis acceptance"))

    p_time = analysis_sub.add_parser("time",
                                     help="Compute calculation time info.")
    p_time.set_defaults(func=lambda args: _not_implemented("analysis time"))

    p_data = analysis_sub.add_parser("data",
                                     help="Parse data from simulation log.")
    mode = p_data.add_mutually_exclusive_group(required=True)
    mode.add_argument("-v", "--vanilla", action="store_true")
    mode.add_argument("-m", "--mbar",    action="store_true")
    p_data.add_argument("--cut", type=int, required=False)
    p_data.set_defaults(func=lambda args: _not_implemented("analysis data"))

    args = parser.parse_args()
    args.func(args)


def _not_implemented(name: str) -> None:
    """Print a stub message for unimplemented analysis sub-commands.

    Args:
        name: Full sub-command name (e.g. ``"analysis acceptance"``).
    """
    print(f"kapybara {name}: not yet implemented in this version.")


if __name__ == "__main__":
    main()
