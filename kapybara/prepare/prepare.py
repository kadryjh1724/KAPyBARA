"""Prepare: prerun LAMMPS simulations (minimize → equilibrate → production).

Replaces the prepare class from prepare/prepare.py + the simulation base
class from prepare/initialize.py. Uses composition:

- core/lammps_setup.py  for LAMMPS instance + Kob-Andersen system setup
- core/thermostat.py    for thermostat strategy
- state/db.py           for completion tracking (replaces .chk files)
"""

import os
import time
import numpy as np

from kapybara.config.schema import SimulationConfig
from kapybara.config.paths import PathManager
from kapybara.state.db import StateDB
from kapybara.core.lammps_setup import create_lammps_instance, setup_kob_andersen
from kapybara.core.thermostat import create_thermostat
from kapybara.core.log_arrays import initialize_log_dict
from kapybara.utils.convert import npy2Cdouble


class Prepare:
    """Prerun LAMMPS workflow: minimize → equilibrate → production.

    Writes per-replica trajectories to step1/trj/{T}/{r:02d}.npy and
    energies to step1/ene/{T}/{r:02d}.npy. Reports completion to StateDB.
    """

    def __init__(self, config: SimulationConfig, paths: PathManager,
                 state_db: StateDB):
        """Initialise the Prepare workflow.

        Args:
            config: Frozen :class:`~kapybara.config.schema.SimulationConfig`.
            paths: :class:`~kapybara.config.paths.PathManager` instance.
            state_db: :class:`~kapybara.state.db.StateDB` instance (queue-mode
                or direct).
        """
        self.config   = config
        self.paths    = paths
        self.state_db = state_db

    # ── Public entry point ────────────────────────────────────────────────

    def prerun(self, T: str, replica_index: int) -> None:
        """Execute the full prerun workflow for one (T, replica) point.

        Runs energy minimization, NVT equilibration, and production MD in
        sequence. Saves the trajectory and energy arrays to
        ``step1/trj/{T}/{replica:02d}.npy`` and
        ``step1/ene/{T}/{replica:02d}.npy``. Reports running/completed/failed
        status to StateDB.

        Args:
            T: Temperature string.
            replica_index: Zero-based replica index.

        Raises:
            Exception: Re-raises any LAMMPS or I/O error after recording
                ``'failed'`` status in StateDB and closing the LAMMPS instance.
        """
        cfg = self.config
        thermostat = create_thermostat(cfg)
        lmp = None

        try:
            self.state_db.update_prerun_state(T, replica_index, "running")

            lmp = create_lammps_instance()
            setup_kob_andersen(lmp, cfg)

            # MSC thermostat needs initial velocities before minimization
            if cfg.thermostat == "MSC":
                vel_start = (np.random.normal(size=(cfg.n_particles, 3))
                             * np.sqrt(float(T)))
                lmp.scatter_atoms("v", 1, 3, npy2Cdouble(vel_start.flatten()))

            t1 = self._run_minimize(lmp)
            t2 = self._run_equilibration(lmp, T, thermostat)
            t3 = self._run_production(lmp, T, thermostat, replica_index)

            print(f"{replica_index:02d},{t1:.2f},{t2:.2f},{t3:.2f}")

            self.state_db.mark_prerun_completed(T, replica_index)

        except Exception as e:
            print(f"Prepare: error prerun T={T} r#{replica_index}: {e}.")
            self.state_db.update_prerun_state(T, replica_index, "failed",
                                              error=str(e))
            raise
        finally:
            if lmp is not None:
                lmp.close()

    # ── LAMMPS phases ─────────────────────────────────────────────────────

    def _run_minimize(self, lmp) -> float:
        """Run energy minimization on the initial random configuration.

        Uses LAMMPS ``minimize`` with tolerances ``1e-4`` (energy) and
        ``1e-6`` (force), up to 10,000 iterations.

        Args:
            lmp: Active LAMMPS instance.

        Returns:
            Wall-clock elapsed time in seconds.
        """
        t0 = time.perf_counter()
        lmp.command("minimize 1.0e-4 1.0e-6 1000 10000")
        return time.perf_counter() - t0

    def _run_equilibration(self, lmp, T: str, thermostat) -> float:
        """Run NVT equilibration to thermalise the system.

        Applies the thermostat, runs for ``nsteps_equil`` steps, then
        removes the thermostat fixes.

        Args:
            lmp: Active LAMMPS instance.
            T: Target temperature string.
            thermostat: :class:`~kapybara.core.thermostat.Thermostat` instance.

        Returns:
            Wall-clock elapsed time in seconds.
        """
        cfg = self.config
        t0 = time.perf_counter()

        lmp.command("reset_timestep 0")
        thermostat.fix(lmp, T)
        lmp.command(f"thermo {cfg.nstout}")
        lmp.command("thermo_style custom step temp ke pe etotal press")
        lmp.command("thermo_modify norm no")
        lmp.command(f"run {cfg.nsteps_equil}")
        thermostat.unfix(lmp)

        return time.perf_counter() - t0

    def _run_production(self, lmp, T: str, thermostat,
                        replica_index: int) -> float:
        """Run production MD and save trajectory and energy arrays.

        Collects positions, velocities, PE, and KE at every ``nstout`` steps
        for ``nloops + 1`` frames (including the initial frame). Saves the
        trajectory as ``(nloops+1, n_particles, 6)`` (pos + vel stacked) and
        energies as ``(2, nloops+1)`` (PE row, KE row).

        Args:
            lmp: Active LAMMPS instance.
            T: Target temperature string.
            thermostat: :class:`~kapybara.core.thermostat.Thermostat` instance.
            replica_index: Zero-based replica index (used for output file names).

        Returns:
            Wall-clock elapsed time in seconds.
        """
        cfg = self.config
        t0 = time.perf_counter()

        log = initialize_log_dict(cfg.nloops, cfg.n_particles)
        log["pos"][0] = np.array(lmp.gather_atoms("x", 1, 3)).reshape(-1, 3)
        log["vel"][0] = np.array(lmp.gather_atoms("v", 1, 3)).reshape(-1, 3)
        log["pe"][0]  = lmp.get_thermo("pe")
        log["ke"][0]  = lmp.get_thermo("ke")

        lmp.command("reset_timestep 0")
        thermostat.fix(lmp, T)
        lmp.command(f"thermo {cfg.nstout}")
        lmp.command("thermo_style custom step temp ke pe etotal press")
        lmp.command("thermo_modify norm no")

        for i in range(cfg.nloops):
            lmp.command(f"run {cfg.nstout}")
            log["pos"][i + 1] = np.array(lmp.gather_atoms("x", 1, 3)).reshape(-1, 3)
            log["vel"][i + 1] = np.array(lmp.gather_atoms("v", 1, 3)).reshape(-1, 3)
            log["pe"][i + 1]  = lmp.get_thermo("pe")
            log["ke"][i + 1]  = lmp.get_thermo("ke")

        thermostat.unfix(lmp)

        trj_path = os.path.join(self.paths.step1_trj, T, f"{replica_index:02d}.npy")
        ene_path = os.path.join(self.paths.step1_ene, T, f"{replica_index:02d}.npy")
        np.save(trj_path, np.concatenate((log["pos"], log["vel"]), axis=2))
        np.save(ene_path, np.vstack((log["pe"], log["ke"])))

        return time.perf_counter() - t0
