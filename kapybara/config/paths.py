"""Path management for all simulation directories and files.

PathManager is the single source of truth for file-system paths. It is
instantiated once from a :class:`~kapybara.config.schema.SimulationConfig`
and passed throughout the package wherever file I/O is needed.
"""

import os

from kapybara.config.schema import SimulationConfig


class PathManager:
    """Constructs and manages all simulation directory and file paths.

    Attributes:
        base: Root directory (``{work_directory}/{job_name}``).
        step1: Prerun output root directory.
        step1_trj: Per-temperature trajectory subdirectory under step1.
        step1_ene: Per-temperature energy subdirectory under step1.
        step1_stdout: SLURM stdout directory for prerun jobs.
        step1_stderr: SLURM stderr directory for prerun jobs.
        step2: TPS output root directory.
        step2_trj: Per-(T, field) trajectory subdirectory under step2.
        step2_ene: Per-(T, field) energy subdirectory under step2.
        step2_csv: Per-(T, field) CSV log subdirectory under step2.
        step2_stdout: SLURM stdout directory for TPS jobs.
        step2_stderr: SLURM stderr directory for TPS jobs.
        step3: Analysis output root directory.
        step3_acc: Acceptance statistics subdirectory under step3.
        step3_dat: Data output subdirectory under step3.
        step3_mbar: MBAR analysis output subdirectory under step3.
        step3_stdout: SLURM stdout directory for analysis jobs.
        step3_stderr: SLURM stderr directory for analysis jobs.
        db: Absolute path to the SQLite state database file.
    """

    def __init__(self, config: SimulationConfig):
        """Build all paths from the simulation configuration.

        Args:
            config: Frozen :class:`~kapybara.config.schema.SimulationConfig`
                instance.
        """
        self.config = config
        self.base = os.path.join(config.work_directory, config.job_name)

        # Step 1: prerun
        self.step1          = os.path.join(self.base, "step1")
        self.step1_trj      = os.path.join(self.step1, "trj")
        self.step1_ene      = os.path.join(self.step1, "ene")
        self.step1_stdout   = os.path.join(self.step1, "stdout")
        self.step1_stderr   = os.path.join(self.step1, "stderr")

        # Step 2: TPS
        self.step2          = os.path.join(self.base, "step2")
        self.step2_trj      = os.path.join(self.step2, "trj")
        self.step2_ene      = os.path.join(self.step2, "ene")
        self.step2_csv      = os.path.join(self.step2, "csv")
        self.step2_stdout   = os.path.join(self.step2, "stdout")
        self.step2_stderr   = os.path.join(self.step2, "stderr")

        # Step 3: analysis
        self.step3          = os.path.join(self.base, "step3")
        self.step3_acc      = os.path.join(self.step3, "acc")
        self.step3_dat      = os.path.join(self.step3, "dat")
        self.step3_mbar     = os.path.join(self.step3, "mbar")
        self.step3_stdout   = os.path.join(self.step3, "stdout")
        self.step3_stderr   = os.path.join(self.step3, "stderr")

        # DB
        self.db             = os.path.join(self.base, "kapybara.db")

    def ensure_directories(self) -> None:
        """Create all required simulation directories (idempotent).

        Creates the full directory tree under ``base``, including
        per-temperature and per-field subdirectories for step1 and step2,
        and all step3 analysis directories. Existing directories are left
        unchanged (``exist_ok=True``).
        """
        cfg = self.config

        os.makedirs(self.base, exist_ok=True)

        # Step 1: per-temperature directories under trj/ and ene/
        for path in (self.step1_trj, self.step1_ene):
            for T in cfg.T:
                os.makedirs(os.path.join(path, T), exist_ok=True)
        for path in (self.step1_stdout, self.step1_stderr):
            os.makedirs(path, exist_ok=True)

        # Step 2: per-temperature + per-field directories
        step2_data_dirs = (self.step2_trj, self.step2_ene, self.step2_csv)
        for T in cfg.T:
            for path in step2_data_dirs:
                if cfg.runtype == "s":
                    for s in cfg.s:
                        os.makedirs(os.path.join(path, T, s), exist_ok=True)
                elif cfg.runtype == "g":
                    for g in cfg.g:
                        os.makedirs(os.path.join(path, T, g), exist_ok=True)
                elif cfg.runtype == "sg":
                    for s in cfg.s:
                        os.makedirs(os.path.join(path, T, s), exist_ok=True)
                        for g in cfg.g:
                            os.makedirs(os.path.join(path, T, s, g), exist_ok=True)
            for path in (self.step2_stdout, self.step2_stderr):
                os.makedirs(os.path.join(path, T), exist_ok=True)

        # Step 3: analysis output directories
        for path in (self.step3_acc, self.step3_dat, self.step3_mbar,
                     self.step3_stdout, self.step3_stderr):
            os.makedirs(path, exist_ok=True)
