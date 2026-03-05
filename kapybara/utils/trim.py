"""CSV trimming utility for TPS restart.

Trims the per-replica CSV log back to a given restart index so that
re-run entries do not duplicate existing records.
"""

import pandas as pd


def trim_csv(csv_path: str, restart_idx: int, run_type: str) -> None:
    """Trim csv_path to the row matching (run_type, restart_idx) inclusive.

    Args:
        csv_path:    Path to the per-replica CSV log file.
        restart_idx: 0-indexed run index to restart from.
        run_type:    'RELAX' or 'ACQUI'.

    Raises:
        ValueError: if no matching row is found.
    """
    df = pd.read_csv(csv_path, dtype={"RUN_IDX": str})
    matches = df[
        (df["RUN_TYPE"] == run_type) & (df["RUN_IDX"] == str(restart_idx))
    ]
    if len(matches) == 0:
        raise ValueError(
            f"trim_csv: no row found for RUN_TYPE={run_type}, "
            f"RUN_IDX={restart_idx} in {csv_path}."
        )
    cut = matches.index[0]
    df.iloc[: cut + 1].to_csv(csv_path, index=False)
