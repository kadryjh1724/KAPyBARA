"""cli/process.py — shared process-lookup utilities for kapybara CLI commands."""

import os
import subprocess


def find_scheduler_pid(config_path: str) -> int | None:
    """Find the PID of the running ``kapybara run`` process for this config.

    Uses ``ps aux | grep`` to locate a process matching the config file name,
    excluding monitor, queue, and stop processes. Returns ``None`` if not found.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Integer PID of the first matching process, or ``None`` if none found.
    """
    config_name = os.path.basename(config_path)
    cmd = (
        f"ps aux | grep 'kapybara.*{config_name}'"
        f" | grep -v grep"
        f" | grep -v 'kapybara monitor'"
        f" | grep -v 'kapybara queue'"
        f" | grep -v 'kapybara stop'"
        f" | awk '{{print $2}}'"
    )
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    pids = [p for p in result.stdout.strip().split("\n") if p]
    return int(pids[0]) if pids else None
