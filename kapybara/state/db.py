"""SQLite-backed central state tracker.

Manages four tables across two layers:

- ``prerun_jobs``  — one row per temperature; tracks SLURM job submission.
- ``prerun_state`` — one row per (T, replica); tracks individual replica status.
- ``tps_jobs``     — one row per (T, field_value); tracks SLURM job submission.
- ``tps_state``    — one row per (T, field_value, replica); tracks individual
  replica phase, run_index, and completion status.

Single-writer architecture: direct :class:`StateDB` instances
(``write_queue=None``) are used by the Scheduler and the
:class:`~kapybara.state.writer.DBWriter` subprocess. Worker processes use
:class:`StateDB` with a ``write_queue`` to enqueue write requests to the
single writer, avoiding SQLite WAL corruption under heavy concurrent writes.
"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone


SCHEMA = """
CREATE TABLE IF NOT EXISTS prerun_jobs (
    T            TEXT    NOT NULL,
    slurm_job_id TEXT,
    status       TEXT    NOT NULL DEFAULT 'pending',
    submitted_at TEXT,
    started_at   TEXT,
    completed_at TEXT,
    PRIMARY KEY (T)
);

CREATE TABLE IF NOT EXISTS prerun_state (
    T            TEXT    NOT NULL,
    replica      INTEGER NOT NULL,
    status       TEXT    NOT NULL DEFAULT 'pending',
    error        TEXT,
    started_at   TEXT,
    completed_at TEXT,
    PRIMARY KEY (T, replica)
);

CREATE TABLE IF NOT EXISTS tps_jobs (
    T            TEXT    NOT NULL,
    field_value  TEXT    NOT NULL,
    dependency   TEXT,
    slurm_job_id TEXT,
    status       TEXT    NOT NULL DEFAULT 'pending',
    submitted_at TEXT,
    started_at   TEXT,
    completed_at TEXT,
    PRIMARY KEY (T, field_value)
);

CREATE TABLE IF NOT EXISTS tps_state (
    T            TEXT    NOT NULL,
    field_value  TEXT    NOT NULL,
    replica      INTEGER NOT NULL,
    phase        TEXT    NOT NULL DEFAULT 'relax',
    run_index    INTEGER NOT NULL DEFAULT 0,
    status       TEXT    NOT NULL DEFAULT 'pending',
    error        TEXT,
    started_at   TEXT,
    completed_at TEXT,
    PRIMARY KEY (T, field_value, replica)
);

CREATE INDEX IF NOT EXISTS idx_prerun_jobs_status
    ON prerun_jobs(status);
CREATE INDEX IF NOT EXISTS idx_prerun_state_status
    ON prerun_state(T, status);
CREATE INDEX IF NOT EXISTS idx_tps_jobs_status
    ON tps_jobs(status);
CREATE INDEX IF NOT EXISTS idx_tps_jobs_dependency
    ON tps_jobs(T, dependency);
CREATE INDEX IF NOT EXISTS idx_tps_state_status
    ON tps_state(T, field_value, status);
CREATE INDEX IF NOT EXISTS idx_tps_state_progress
    ON tps_state(T, field_value, phase, run_index);
"""


def _now() -> str:
    """Current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


class StateDB:
    """SQLite-based central state tracker. Located at ``{base}/kapybara.db``.

    Manages four tables in two layers:

    - ``prerun_jobs`` / ``prerun_state``  (job-level + replica-level for prerun)
    - ``tps_jobs``    / ``tps_state``     (job-level + replica-level for TPS)

    In *direct mode* (``write_queue=None``), all methods write to SQLite
    directly. In *queue mode* (``write_queue`` provided), write methods
    enqueue ``(method_name, args)`` tuples for the
    :class:`~kapybara.state.writer.DBWriter` subprocess to execute.
    Read methods always query SQLite directly regardless of mode.

    Attributes:
        _db_path: Absolute path to the SQLite database file.
        _write_queue: Multiprocessing queue for worker-mode writes, or
            ``None`` for direct-mode writes.
    """

    def __init__(self, db_path: str, write_queue=None):
        """Initialise the state tracker.

        In direct mode (``write_queue=None``), the database schema is
        created if it does not yet exist. In queue mode, schema creation is
        assumed to have been done by the main process before workers start.

        Args:
            db_path: Absolute path to the SQLite database file.
            write_queue: Multiprocessing queue shared with a
                :class:`~kapybara.state.writer.DBWriter` subprocess, or
                ``None`` for direct writes.
        """
        self._db_path = db_path
        self._write_queue = write_queue
        if write_queue is None:
            self._init_db()

    def _enqueue(self, method: str, *args) -> None:
        """Put a write request on the DBWriter queue (worker-mode only).

        Args:
            method: Name of the :class:`StateDB` method to call.
            *args: Positional arguments to pass to that method.
        """
        self._write_queue.put((method, args))

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=DELETE")
            conn.execute("PRAGMA synchronous=FULL")
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self._db_path, timeout=60)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=30000")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ══════════════════════════════════════
    # Prerun Job Layer
    # ══════════════════════════════════════

    def register_prerun_job(self, T: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO prerun_jobs (T) VALUES (?)",
                (T,),
            )

    def submit_prerun_job(self, T: str, slurm_job_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """UPDATE prerun_jobs
                   SET slurm_job_id=?, status='submitted', submitted_at=?
                   WHERE T=?""",
                (slurm_job_id, _now(), T),
            )

    def update_prerun_job_status(self, T: str, status: str) -> None:
        with self._connect() as conn:
            if status == "running":
                conn.execute(
                    "UPDATE prerun_jobs SET status=?, started_at=? WHERE T=?",
                    (status, _now(), T),
                )
            elif status == "completed":
                conn.execute(
                    "UPDATE prerun_jobs SET status=?, completed_at=? WHERE T=?",
                    (status, _now(), T),
                )
            else:
                conn.execute(
                    "UPDATE prerun_jobs SET status=? WHERE T=?",
                    (status, T),
                )

    def get_prerun_job(self, T: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM prerun_jobs WHERE T=?", (T,)
            ).fetchone()
            return dict(row) if row else None

    # ══════════════════════════════════════
    # Prerun Replica Layer
    # ══════════════════════════════════════

    def update_prerun_state(self, T: str, replica: int,
                            status: str, error: str = None) -> None:
        if self._write_queue is not None:
            self._enqueue("update_prerun_state", T, replica, status, error)
            return
        now = _now()
        started = now if status == "running" else None
        completed = now if status == "completed" else None
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO prerun_state (T, replica, status, error, started_at, completed_at)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(T, replica) DO UPDATE SET
                       status=excluded.status,
                       error=excluded.error,
                       started_at=COALESCE(prerun_state.started_at, excluded.started_at),
                       completed_at=COALESCE(excluded.completed_at, prerun_state.completed_at)""",
                (T, replica, status, error, started, completed),
            )

    def mark_prerun_completed(self, T: str, replica: int) -> None:
        if self._write_queue is not None:
            self._enqueue("mark_prerun_completed", T, replica)
            return
        self.update_prerun_state(T, replica, "completed")

    def is_prerun_completed(self, T: str, n_replica: int) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM prerun_state WHERE T=? AND status='completed'",
                (T,),
            ).fetchone()
            return row["cnt"] >= n_replica

    def all_preruns_completed(self, temperatures: list[str],
                              n_replica: int) -> bool:
        return all(
            self.is_prerun_completed(T, n_replica) for T in temperatures
        )

    def get_prerun_statuses(self) -> dict[str, str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT T, status FROM prerun_jobs"
            ).fetchall()
            return {row["T"]: row["status"] for row in rows}

    # ══════════════════════════════════════
    # TPS Job Layer
    # ══════════════════════════════════════

    def register_tps_job(self, T: str, field_value: str,
                         dependency: str | None) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO tps_jobs (T, field_value, dependency)
                   VALUES (?, ?, ?)""",
                (T, field_value, dependency),
            )

    def submit_tps_job(self, T: str, field_value: str,
                       slurm_job_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """UPDATE tps_jobs
                   SET slurm_job_id=?, status='submitted', submitted_at=?
                   WHERE T=? AND field_value=?""",
                (slurm_job_id, _now(), T, field_value),
            )

    def update_tps_job_status(self, T: str, field_value: str,
                              status: str) -> None:
        with self._connect() as conn:
            if status == "running":
                conn.execute(
                    """UPDATE tps_jobs SET status=?, started_at=?
                       WHERE T=? AND field_value=?""",
                    (status, _now(), T, field_value),
                )
            elif status == "completed":
                conn.execute(
                    """UPDATE tps_jobs SET status=?, completed_at=?
                       WHERE T=? AND field_value=?""",
                    (status, _now(), T, field_value),
                )
            else:
                conn.execute(
                    """UPDATE tps_jobs SET status=?
                       WHERE T=? AND field_value=?""",
                    (status, T, field_value),
                )

    def get_tps_job(self, T: str, field_value: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM tps_jobs WHERE T=? AND field_value=?",
                (T, field_value),
            ).fetchone()
            return dict(row) if row else None

    def get_dependency(self, T: str, field_value: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT dependency FROM tps_jobs WHERE T=? AND field_value=?",
                (T, field_value),
            ).fetchone()
            return row["dependency"] if row else None

    # ══════════════════════════════════════
    # TPS Replica Layer
    # ══════════════════════════════════════

    def update_tps_state(self, T: str, field_value: str, replica: int,
                         phase: str, run_index: int,
                         status: str, error: str = None) -> None:
        if self._write_queue is not None:
            self._enqueue("update_tps_state", T, field_value, replica,
                          phase, run_index, status, error)
            return
        now = _now()
        started = now if status == "running" else None
        completed = now if status == "completed" else None
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO tps_state
                       (T, field_value, replica, phase, run_index, status, error,
                        started_at, completed_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(T, field_value, replica) DO UPDATE SET
                       phase=excluded.phase,
                       run_index=excluded.run_index,
                       status=excluded.status,
                       error=excluded.error,
                       started_at=COALESCE(tps_state.started_at, excluded.started_at),
                       completed_at=COALESCE(excluded.completed_at, tps_state.completed_at)""",
                (T, field_value, replica, phase, run_index, status, error,
                 started, completed),
            )

    def mark_tps_completed(self, T: str, field_value: str,
                           replica: int) -> None:
        if self._write_queue is not None:
            self._enqueue("mark_tps_completed", T, field_value, replica)
            return
        now = _now()
        with self._connect() as conn:
            conn.execute(
                """UPDATE tps_state SET status='completed', completed_at=?
                   WHERE T=? AND field_value=? AND replica=?""",
                (now, T, field_value, replica),
            )
            # Check if all replicas for this job are completed
            row = conn.execute(
                """SELECT COUNT(*) AS total,
                          SUM(status='completed') AS done
                   FROM tps_state
                   WHERE T=? AND field_value=?""",
                (T, field_value),
            ).fetchone()
            if row["total"] > 0 and row["total"] == row["done"]:
                conn.execute(
                    """UPDATE tps_jobs SET status='completed', completed_at=?
                       WHERE T=? AND field_value=?""",
                    (now, T, field_value),
                )

    # ══════════════════════════════════════
    # TPS Read (scheduler, monitor)
    # ══════════════════════════════════════

    def get_tps_job_status(self, T: str, field_value: str) -> str:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT status FROM tps_jobs WHERE T=? AND field_value=?",
                (T, field_value),
            ).fetchone()
            return row["status"] if row else "pending"

    def get_all_tps_statuses(self) -> dict[str, dict[str, str]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT T, field_value, status FROM tps_jobs"
            ).fetchall()
            result: dict[str, dict[str, str]] = {}
            for row in rows:
                result.setdefault(row["T"], {})[row["field_value"]] = row["status"]
            return result

    def get_tps_replica_progress(self, T: str,
                                 field_value: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT replica, phase, run_index, status
                   FROM tps_state
                   WHERE T=? AND field_value=?
                   ORDER BY replica""",
                (T, field_value),
            ).fetchall()
            return [dict(row) for row in rows]

    def can_branch_from(self, T: str, field_value: str,
                        n_branch: int, n_replica: int) -> bool:
        """Check whether a running parent job has accumulated enough acquisitions.

        A child job may branch from a parent once all ``n_replica`` replicas
        of the parent have recorded ``run_index >= n_branch`` in the
        ``acqui`` phase.

        Args:
            T: Temperature string.
            field_value: Parent field value string.
            n_branch: Minimum acquisition run index required.
            n_replica: Total number of replicas required to satisfy the
                branching condition.

        Returns:
            ``True`` if all replicas of (T, field_value) meet the branching
            threshold.
        """
        with self._connect() as conn:
            row = conn.execute(
                """SELECT COUNT(*) AS cnt FROM tps_state
                   WHERE T=? AND field_value=?
                     AND phase='acqui' AND run_index >= ?""",
                (T, field_value, n_branch),
            ).fetchone()
            return row["cnt"] >= n_replica

    # ══════════════════════════════════════
    # SLURM Sync (scheduler)
    # ══════════════════════════════════════

    def get_running_prerun_jobs(self) -> dict[str, str]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT T, slurm_job_id FROM prerun_jobs
                   WHERE status IN ('submitted', 'running')"""
            ).fetchall()
            return {row["T"]: row["slurm_job_id"] for row in rows}

    def get_running_tps_jobs(self) -> dict[tuple[str, str], str]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT T, field_value, slurm_job_id FROM tps_jobs
                   WHERE status IN ('submitted', 'running')"""
            ).fetchall()
            return {
                (row["T"], row["field_value"]): row["slurm_job_id"]
                for row in rows
            }

    def mark_missing_prerun_failed(self, T: str) -> None:
        now = _now()
        with self._connect() as conn:
            conn.execute(
                """UPDATE prerun_state SET status='failed', error='SLURM job disappeared'
                   WHERE T=? AND status IN ('pending', 'running')""",
                (T,),
            )
            conn.execute(
                "UPDATE prerun_jobs SET status='failed' WHERE T=?",
                (T,),
            )

    def mark_missing_tps_failed(self, T: str, field_value: str) -> None:
        now = _now()
        with self._connect() as conn:
            conn.execute(
                """UPDATE tps_state SET status='failed', error='SLURM job disappeared'
                   WHERE T=? AND field_value=? AND status IN ('pending', 'running')""",
                (T, field_value),
            )
            conn.execute(
                """UPDATE tps_jobs SET status='failed'
                   WHERE T=? AND field_value=?""",
                (T, field_value),
            )

    # ══════════════════════════════════════
    # Per-Replica Completion Checks (workers / CLI)
    # ══════════════════════════════════════

    def get_pending_prerun_replicas(self, T: str, n_replica: int) -> list[int]:
        """Return prerun replica indices that have not yet completed.

        Used by ``kapybara prerun`` to determine which replicas still need to
        be submitted (pending, failed, or never started).

        Args:
            T: Temperature string.
            n_replica: Total expected number of replicas.

        Returns:
            Sorted list of replica indices in ``range(n_replica)`` that are
            not in ``'completed'`` status.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT replica FROM prerun_state WHERE T=? AND status='completed'",
                (T,),
            ).fetchall()
        completed = {row["replica"] for row in rows}
        return [r for r in range(n_replica) if r not in completed]

    def get_pending_tps_replicas(self, T: str, field_value: str,
                                  n_replica: int) -> list[int]:
        """Return TPS replica indices that have not yet completed.

        Used by the Scheduler to allocate only the CPUs needed for
        unfinished replicas when resubmitting a failed job.

        Args:
            T: Temperature string.
            field_value: Field value string.
            n_replica: Total expected number of replicas.

        Returns:
            Sorted list of replica indices in ``range(n_replica)`` that are
            not in ``'completed'`` status.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT replica FROM tps_state
                   WHERE T=? AND field_value=? AND status='completed'""",
                (T, field_value),
            ).fetchall()
        completed = {row["replica"] for row in rows}
        return [r for r in range(n_replica) if r not in completed]



    def is_prerun_replica_completed(self, T: str, replica: int) -> bool:
        """Check if a single prerun replica is already completed."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT status FROM prerun_state WHERE T=? AND replica=?",
                (T, replica),
            ).fetchone()
            return row is not None and row["status"] == "completed"

    def is_tps_replica_completed(self, T: str, field_value: str,
                                  replica: int) -> bool:
        """Check if a single TPS replica is already completed."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT status FROM tps_state WHERE T=? AND field_value=? AND replica=?",
                (T, field_value, replica),
            ).fetchone()
            return row is not None and row["status"] == "completed"
