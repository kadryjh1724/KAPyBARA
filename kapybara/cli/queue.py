"""cli/queue.py — 'kapybara queue' handler.

Modified squeue display optimised for KAPyBARA. Replaces per-replica
CSV line-count progress with a StateDB query.

Column widths for T and field are derived from config.n_decimals so the
table layout adapts automatically to the actual value string lengths.
"""

import os
import sys
import time
import signal
import subprocess

import matplotlib.cm as cm

from kapybara.config.loader import load_config
from kapybara.config.paths import PathManager
from kapybara.state.db import StateDB
from kapybara.utils.convert import strfdelta2


def _signal_handler(sig, frame):
    """Handle SIGINT by printing a termination message and exiting cleanly."""
    print("\nKapyqueue terminated.")
    sys.exit(0)


def _parse_time_string(time_str: str) -> int:
    """Convert SLURM time string (D-HH:MM:SS or HH:MM:SS or MM:SS) to seconds."""
    if "-" in time_str:
        day_part, time_part = time_str.split("-")
        days = int(day_part) * 86400
        h, m, s = (int(x) for x in time_part.split(":"))
        return days + h * 3600 + m * 60 + s
    parts = [int(x) for x in time_str.split(":")]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    raise ValueError(f"Unrecognised time format: {time_str}")


def _get_pid(config_path: str) -> str:
    """Find the PID of the running ``kapybara run`` process for this config.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        PID string of the first matching process, or ``"N/A"`` if none found.
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
    return pids[0] if pids else "N/A"


def _progress_color(progress: int, total: int, colormap: str = "RdYlGn") -> str:
    """Return a terminal-coloured 3-character percentage string."""
    cmap = cm.get_cmap(colormap)
    ratio = min(progress / total, 1.0) if total > 0 else 0.0
    r, g, b, _ = cmap(ratio)
    ri, gi, bi = int(r * 255), int(g * 255), int(b * 255)
    brightness = (0.299 * ri + 0.587 * gi + 0.114 * bi) / 255
    text = "\033[30m" if brightness > 0.5 else "\033[37m"
    pct = int(ratio * 100)
    return f"\033[48;2;{ri};{gi};{bi}m{text}{pct:03d}\033[0m"


def _job_progress(state_db: StateDB, T: str, field_value: str,
                  n_relax: int) -> int:
    """Compute the approximate number of completed TPS steps for one job.

    Sums per-replica progress counts from StateDB. Completed replicas
    contribute ``n_relax + n_acqui`` steps; in-progress replicas contribute
    their current ``run_index``, offset by ``n_relax`` if in the acqui phase.

    Args:
        state_db: :class:`~kapybara.state.db.StateDB` instance.
        T: Temperature string.
        field_value: Field value string.
        n_relax: Total number of relax steps per replica.

    Returns:
        Summed step count across all replicas for this job.
    """
    rows = state_db.get_tps_replica_progress(T, field_value)
    total = 0
    for row in rows:
        if row["status"] == "completed":
            total += n_relax + row["run_index"]
        elif row["phase"] == "acqui":
            total += n_relax + row["run_index"]
        else:
            total += row["run_index"]
    return total


def _colorbar_line(width: int) -> str:
    """Build the RdYlGn colour-gradient header line for the progress bar.

    Renders a continuous colour gradient from red (0%) on the right to green
    (100%) on the left, with percentage labels at 0%, 25%, 50%, 75%, 100%.

    Args:
        width: Content width in characters (excluding box-drawing borders).

    Returns:
        A terminal-escape-coded string including the surrounding ``║ ... ║``
        box-drawing borders.
    """
    cmap = cm.get_cmap("RdYlGn")
    labels = {
        0:                "100",
        width // 4:       "75",
        width // 2:       "50",
        3 * width // 4:   "25",
        width - 1:        "0",
    }
    line = "║ "
    i = 0
    while i < width:
        ratio = (width - 1 - i) / (width - 1)
        r, g, b, _ = cmap(ratio)
        ri, gi, bi = int(r * 255), int(g * 255), int(b * 255)
        label = labels.get(i)
        if label:
            line += f"\033[48;2;{ri};{gi};{bi}m\033[30m{label}\033[0m"
            i += len(label)
        else:
            line += f"\033[48;2;{ri};{gi};{bi}m \033[0m"
            i += 1
    line += " ║"
    return line


def queue(args) -> None:
    """Handle the ``kapybara queue`` sub-command.

    Queries ``squeue`` for jobs matching this simulation's job name, then
    renders a box-drawing table with per-job runtime, node, and colour-coded
    progress. PENDING jobs show ``-`` placeholders instead of live data.
    In watch mode, refreshes every ``args.watch`` seconds until Ctrl+C.

    Args:
        args: Parsed :class:`argparse.Namespace` with attributes:
            ``config`` (str), ``watch`` (int or None), ``number`` (int).
    """
    config   = load_config(args.config, quiet=True)
    paths    = PathManager(config)
    state_db = StateDB(paths.db)

    if config.runtype not in ("g", "s"):
        print(f"kapybara queue: not implemented for runtype={config.runtype}.")
        return

    field_label = config.runtype
    f_vals      = config.g if config.runtype == "g" else config.s
    n_T         = len(config.T)
    n_field     = len(f_vals)
    total       = config.n_replica * (config.n_relax + config.n_acqui)

    # ── Column widths derived from actual formatted value lengths ──────────
    # config.T / config.g / config.s are already formatted strings (e.g. "0.6500")
    T_val_w = max(len(t) for t in config.T)
    f_val_w = max(len(f) for f in f_vals)

    # Column width = value width + 1 space on each side; at least wide enough
    # to display the column header label.
    T_col_w = max(T_val_w + 2, len("T") + 2)
    f_col_w = max(f_val_w + 2, len(field_label) + 2)

    # Fixed column widths (number of characters between ║ marks)
    IDX_W   = 9    # "  INDEX  "
    NTYPE_W = 10   # " NODETYPE "
    RT_W    = 17   # " DDd HHh MMm SSs "  (strfdelta2 = 15 chars)
    NODE_W  = 8    # "  NNNNNN  "
    PCT_W   = 5    # "  NNN  "  (_progress_color = 3 chars)

    # Total inner width: sum of column widths + one ║ separator between each
    _cols = [IDX_W, NTYPE_W, T_col_w, f_col_w, RT_W, NODE_W, PCT_W]
    inner = sum(_cols) + len(_cols) - 1   # 6 separators for 7 columns

    # Info-header section: left column is fixed at 20, right takes the rest
    INFO_L   = 20
    INFO_RW  = inner - INFO_L - 1    # right column width (−1 for ╦/╩)
    INFO_RCW = INFO_RW - 1           # right content width (−1 for leading " " in "║ ")

    # ── Box-drawing helpers ────────────────────────────────────────────────

    def _hline(lc: str, mc: str, rc: str) -> str:
        """Horizontal divider with left/mid/right corner characters."""
        return lc + mc.join("═" * w for w in _cols) + rc

    def _top() -> str:
        tag = "[ KAPY-QUEUE ]"
        return f"╔{'═' * INFO_L}╦{tag.rjust(INFO_RW, '═')}╗"

    def _info(label: str, value: str) -> str:
        return "║" + f" {label}".ljust(INFO_L) + "║ " + value.ljust(INFO_RCW) + "║"

    def _progress_hdr() -> str:
        tag = "[ PROGRESS BAR ]"
        return f"╠{'═' * INFO_L}╩{tag.rjust(INFO_RW, '═')}╣"

    def _col_header() -> str:
        cells = [
            "INDEX".center(IDX_W),
            "NODETYPE".center(NTYPE_W),
            "T".center(T_col_w),
            field_label.upper().center(f_col_w),
            "RUNTIME".center(RT_W),
            "NODE".center(NODE_W),
            "%".center(PCT_W),
        ]
        return "║" + "║".join(cells) + "║"

    def _data_row(job_id: str, partition: str, T: str, field_value: str,
                  dt_secs: int, node: str, progress: int,
                  is_pending: bool = False) -> str:
        if is_pending:
            rt  = "-".center(RT_W - 2)
            nd  = "-".center(NODE_W - 2)
            pct = " - "
        else:
            rt  = strfdelta2(dt_secs)
            nd  = node[:NODE_W - 2].center(NODE_W - 2)
            pct = _progress_color(progress, total)
        return (
            "║ " + job_id.rjust(IDX_W - 2)
            + " ║ " + partition.center(NTYPE_W - 2)
            + " ║ " + T.center(T_col_w - 2)
            + " ║ " + field_value.rjust(f_col_w - 2)
            + " ║ " + rt
            + " ║ " + nd
            + " ║ " + pct
            + " ║"
        )

    # ── Main loop ─────────────────────────────────────────────────────────

    while True:
        if args.watch:
            signal.signal(signal.SIGINT, _signal_handler)
            os.system("clear")

        cmd = (
            f"squeue --format \"%i %P %j %M %R\" -u $(whoami) --sort=i"
            f" | grep {config.job_name}"
            f" | grep -v prerun"
        )
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        lines = [l for l in result.stdout.strip().split("\n") if l.strip()]

        print(_top())
        print(_info("JOB NAME",      config.job_name))
        print(_info("PROCESS PID",   _get_pid(args.config)))
        print(_info(f"(T, {field_label}) DIM", f"({n_T}, {n_field})"))
        print(_info("NUM. REPLICAS", str(config.n_replica)))
        print(_progress_hdr())
        print(_colorbar_line(inner - 2))
        print(_hline("╠", "╦", "╣"))
        print(_col_header())
        print(_hline("╠", "╬", "╣"))

        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                continue
            job_id, partition, job_name, dt_str, node = parts[:5]
            name_parts  = job_name.split("_")
            T           = name_parts[-2]
            field_value = name_parts[-1]
            dt_secs     = _parse_time_string(dt_str)

            if partition in ("smallmem,largemem",):
                partition = "PENDING"

            is_pending = node.startswith("(")
            if is_pending:
                node = "-"

            progress = 0 if is_pending else _job_progress(state_db, T, field_value, config.n_relax)
            print(_data_row(job_id, partition, T, field_value, dt_secs, node, progress, is_pending))

        print(_hline("╚", "╩", "╝"))

        if args.watch is None:
            break
        time.sleep(args.watch)
