"""Dedicated single-writer process for SQLite state database.

Architecture:
    worker processes
        → multiprocessing.Queue
            → DBWriter (one process)
                → StateDB (direct write)

This ensures only one process ever writes to SQLite, eliminating WAL
corruption that occurs under heavy concurrent multiprocessing writes.
"""

import sys
import multiprocessing as mp
from multiprocessing import Queue as MPQueue


def _writer_loop(db_path: str, queue: MPQueue) -> None:
    """Writer subprocess event loop.

    Blocks on ``queue.get()``, dispatching each ``(method_name, args)``
    tuple to the corresponding :class:`~kapybara.state.db.StateDB` method.
    Terminates when it receives ``None`` (the shutdown sentinel).

    Args:
        db_path: Absolute path to the SQLite database file.
        queue: Multiprocessing queue shared with the parent process and all
            worker processes.
    """
    from kapybara.state.db import StateDB

    db = StateDB(db_path)  # direct mode — no write_queue
    while True:
        msg = queue.get()   # blocks until a message arrives
        if msg is None:     # None is the shutdown sentinel
            break
        method_name, args = msg
        try:
            getattr(db, method_name)(*args)
        except Exception as e:
            print(f"[DBWriter] ERROR {method_name}{args}: {e}", file=sys.stderr)


class DBWriter:
    """Manages a dedicated SQLite writer subprocess.

    Usage in the __main__ block of a command module:

        writer = DBWriter(db_path)
        writer.start()
        # pass writer.queue to StateDB(..., write_queue=writer.queue)
        # ... spawn and join workers ...
        writer.stop()
    """

    def __init__(self, db_path: str):
        self._db_path = db_path
        self.queue: MPQueue = mp.Queue()
        self._process: mp.Process | None = None

    def start(self) -> None:
        """Spawn the writer subprocess."""
        self._process = mp.Process(
            target=_writer_loop,
            args=(self._db_path, self.queue),
            daemon=True,
            name="kapybara-db-writer",
        )
        self._process.start()

    def stop(self) -> None:
        """Send the shutdown sentinel and wait for the writer to finish."""
        self.queue.put(None)
        if self._process is not None:
            self._process.join(timeout=60)
            if self._process.is_alive():
                self._process.terminate()
