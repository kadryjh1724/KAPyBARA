"""cli/stop.py — 'kapybara stop' handler.

Sends SIGTERM to the running scheduler process identified by the config file.
Waits up to 5 seconds for clean shutdown; falls back to SIGKILL with --force.
"""

import os
import sys
import time
import signal

from kapybara.cli.process import find_scheduler_pid


def stop(args) -> None:
    """Handle the ``kapybara stop`` sub-command.

    Locates the running ``kapybara run`` scheduler process for the given
    config, sends SIGTERM, and waits up to 5 seconds for it to exit. If the
    process is still alive after 5 seconds, prints a message or (with
    ``--force``) sends SIGKILL.

    Args:
        args: Parsed :class:`argparse.Namespace` with attributes:
            ``config`` (str), ``force`` (bool).
    """
    pid = find_scheduler_pid(args.config)

    if pid is None:
        print("kapybara: no running scheduler found for this config.")
        sys.exit(1)

    os.kill(pid, signal.SIGTERM)
    print(f"kapybara: sent SIGTERM to PID {pid}.")

    for _ in range(10):
        time.sleep(0.5)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            print("kapybara: scheduler stopped.")
            return

    if args.force:
        os.kill(pid, signal.SIGKILL)
        print("kapybara: sent SIGKILL.")
    else:
        print("kapybara: process did not exit. Use --force to send SIGKILL.")
