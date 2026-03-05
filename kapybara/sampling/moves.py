"""TPSMoves: shooting and shifting TPS move implementations.

Holds the current trajectory state and applies Metropolis-Hastings
moves. LAMMPS handle is passed as an argument (dependency injection)
rather than stored as an attribute — the caller (RunnerTg, etc.)
owns the LAMMPS instance.
"""

import numpy as np

from kapybara.config.schema import SimulationConfig
from kapybara.core.thermostat import Thermostat
from kapybara.core.activity import compute_activity
from kapybara.core.log_arrays import initialize_log_dict
from kapybara.utils.convert import npy2Cdouble


class TPSMoves:
    """Shooting/shifting TPS moves with Metropolis-Hastings acceptance.

    The current trajectory is stored as instance state:
        TPS_pos, TPS_vel, TPS_PE, TPS_KE, TPS_K

    The caller must set these before calling any move method (e.g., by
    loading from a .npy file). The move methods update these in-place on
    acceptance.

    Move dispatch is determined once at construction:
        self.shoot  →  run_one_way_shooting  or  run_two_way_shooting
        self.shift  →  run_one_way_shifting  or  run_two_way_shifting
    """

    def __init__(self, config: SimulationConfig, thermostat: Thermostat):
        """Initialise TPS moves and dispatch tables.

        Selects the one-way or two-way variant of each move type based on
        ``config.one_way_shoot`` and ``config.one_way_shift``. The current
        trajectory state attributes (``TPS_pos``, etc.) are set to placeholder
        values; the caller must load a valid trajectory before calling any move.

        Args:
            config: Frozen :class:`~kapybara.config.schema.SimulationConfig`.
            thermostat: :class:`~kapybara.core.thermostat.Thermostat` strategy
                instance.
        """
        self.config = config
        self.thermostat = thermostat

        self.shoot = (self.run_one_way_shooting if config.one_way_shoot
                      else self.run_two_way_shooting)
        self.shift = (self.run_one_way_shifting if config.one_way_shift
                      else self.run_two_way_shifting)

        # Current trajectory state — caller must initialise before first move.
        self.TPS_pos: np.ndarray = None
        self.TPS_vel: np.ndarray = None
        self.TPS_PE:  np.ndarray = None
        self.TPS_KE:  np.ndarray = None
        self.TPS_K:   int        = 0

    # ── Shooting ──────────────────────────────────────────────────────────

    def run_one_way_shooting(self, lmp, pt: int,
                              T: str, s: str, g: str) -> tuple:
        """One-way shooting move: resample velocities at ``pt``, forward run only.

        Draws new Maxwell-Boltzmann velocities at the shooting point and
        integrates forward for the full observation window. Accepts or rejects
        via :meth:`_accept_or_reject`.

        Args:
            lmp: Active LAMMPS instance.
            pt: Shooting point (frame index, 0-indexed).
            T: Temperature string.
            s: s-field value string.
            g: g-field value string.

        Returns:
            Tuple from :meth:`_accept_or_reject`:
            ``(r, cur_E, cur_K, new_E, new_K, dE, dK, factor, accepted)``.
        """
        cfg = self.config
        log = initialize_log_dict(cfg.nloops, cfg.n_particles)

        pos_start = self.TPS_pos[pt]
        vel_start = np.random.normal(size=(cfg.n_particles, 3)) * np.sqrt(float(T))
        log["pos"][0] = pos_start
        log["vel"][0] = vel_start
        log["pe"][0]  = self.TPS_PE[pt]
        log["ke"][0]  = 0.5 * np.sum(vel_start * vel_start)

        lmp.command("reset_timestep 0")
        self.thermostat.fix(lmp, T)
        lmp.command(f"thermo {cfg.nstout}")
        lmp.command("thermo_style custom step temp ke pe etotal press")
        lmp.command("thermo_modify norm no")
        lmp.scatter_atoms("x", 1, 3, npy2Cdouble(pos_start.flatten()))
        lmp.scatter_atoms("v", 1, 3, npy2Cdouble(vel_start.flatten()))

        for i in range(cfg.nloops):
            lmp.command(f"run {cfg.nstout}")
            log["pos"][i + 1] = np.array(lmp.gather_atoms("x", 1, 3)).reshape(-1, 3)
            log["vel"][i + 1] = np.array(lmp.gather_atoms("v", 1, 3)).reshape(-1, 3)
            log["pe"][i + 1]  = lmp.get_thermo("pe")
            log["ke"][i + 1]  = lmp.get_thermo("ke")

        self.thermostat.unfix(lmp)

        return self._accept_or_reject(log, s, g)

    def run_two_way_shooting(self, lmp, pt: int,
                              T: str, s: str, g: str) -> tuple:
        """Two-way shooting move: backward from ``pt``, then forward from ``pt``.

        Draws new Maxwell-Boltzmann velocities, integrates backward (with
        reversed velocities) to fill frames ``[0, pt)``, then forward from
        ``pt`` to fill frames ``(pt, nloops]``.

        Args:
            lmp: Active LAMMPS instance.
            pt: Shooting point (frame index, 0-indexed).
            T: Temperature string.
            s: s-field value string.
            g: g-field value string.

        Returns:
            Tuple from :meth:`_accept_or_reject`.
        """
        cfg = self.config
        log = initialize_log_dict(cfg.nloops, cfg.n_particles)

        pos_start = self.TPS_pos[pt]
        vel_start = np.random.normal(size=(cfg.n_particles, 3)) * np.sqrt(float(T))
        log["pos"][pt] = pos_start
        log["vel"][pt] = vel_start
        log["pe"][pt]  = self.TPS_PE[pt]
        log["ke"][pt]  = 0.5 * np.sum(vel_start * vel_start)

        lmp.command("reset_timestep 0")
        self.thermostat.fix(lmp, T)
        lmp.command(f"thermo {cfg.nstout}")
        lmp.command("thermo_style custom step temp ke pe etotal press")
        lmp.command("thermo_modify norm no")

        # Backward leg (reversed velocities)
        lmp.scatter_atoms("x", 1, 3, npy2Cdouble(pos_start.flatten()))
        lmp.scatter_atoms("v", 1, 3, npy2Cdouble((-vel_start).flatten()))
        for i in range(pt):
            lmp.command(f"run {cfg.nstout}")
            log["pos"][pt - i - 1] = np.array(lmp.gather_atoms("x", 1, 3)).reshape(-1, 3)
            log["vel"][pt - i - 1] = np.array(lmp.gather_atoms("v", 1, 3)).reshape(-1, 3)
            log["pe"][pt - i - 1]  = lmp.get_thermo("pe")
            log["ke"][pt - i - 1]  = lmp.get_thermo("ke")

        # Forward leg
        lmp.scatter_atoms("x", 1, 3, npy2Cdouble(pos_start.flatten()))
        lmp.scatter_atoms("v", 1, 3, npy2Cdouble(vel_start.flatten()))
        for i in range(cfg.nloops - pt):
            lmp.command(f"run {cfg.nstout}")
            log["pos"][pt + i + 1] = np.array(lmp.gather_atoms("x", 1, 3)).reshape(-1, 3)
            log["vel"][pt + i + 1] = np.array(lmp.gather_atoms("v", 1, 3)).reshape(-1, 3)
            log["pe"][pt + i + 1]  = lmp.get_thermo("pe")
            log["ke"][pt + i + 1]  = lmp.get_thermo("ke")

        self.thermostat.unfix(lmp)

        return self._accept_or_reject(log, s, g)

    # ── Shifting ──────────────────────────────────────────────────────────

    def run_one_way_shifting(self, lmp, pt: int,
                              T: str, s: str, g: str) -> tuple:
        """One-way shifting move: shift trajectory forward by ``pt`` frames.

        Copies frames ``[pt, nloops]`` from the current trajectory to the
        start of the new trajectory, then integrates forward from the last
        frame to fill the remaining ``pt`` frames.

        Args:
            lmp: Active LAMMPS instance.
            pt: Shift amount in frames (1-indexed within the trajectory).
            T: Temperature string.
            s: s-field value string.
            g: g-field value string.

        Returns:
            Tuple from :meth:`_accept_or_reject`.
        """
        cfg = self.config
        log = initialize_log_dict(cfg.nloops, cfg.n_particles)

        pos_start = self.TPS_pos[-1]
        vel_start = self.TPS_vel[-1]
        log["pos"][: cfg.nloops - pt + 1] = self.TPS_pos[pt:]
        log["vel"][: cfg.nloops - pt + 1] = self.TPS_vel[pt:]
        log["pe"][: cfg.nloops - pt + 1]  = self.TPS_PE[pt:]
        log["ke"][: cfg.nloops - pt + 1]  = self.TPS_KE[pt:]

        lmp.command("reset_timestep 0")
        self.thermostat.fix(lmp, T)
        lmp.command(f"thermo {cfg.nstout}")
        lmp.command("thermo_style custom step temp ke pe etotal press")
        lmp.command("thermo_modify norm no")
        lmp.scatter_atoms("x", 1, 3, npy2Cdouble(pos_start.flatten()))
        lmp.scatter_atoms("v", 1, 3, npy2Cdouble(vel_start.flatten()))

        for i in range(pt):
            lmp.command(f"run {cfg.nstout}")
            log["pos"][cfg.nloops - pt + i + 1] = np.array(lmp.gather_atoms("x", 1, 3)).reshape(-1, 3)
            log["vel"][cfg.nloops - pt + i + 1] = np.array(lmp.gather_atoms("v", 1, 3)).reshape(-1, 3)
            log["pe"][cfg.nloops - pt + i + 1]  = lmp.get_thermo("pe")
            log["ke"][cfg.nloops - pt + i + 1]  = lmp.get_thermo("ke")

        self.thermostat.unfix(lmp)

        return self._accept_or_reject(log, s, g)

    def run_two_way_shifting(self, lmp, pt: int,
                              T: str, s: str, g: str) -> tuple:
        """Two-way shifting move: forward if ``pt < nloops/2``, backward otherwise.

        If shifting forward, extends from the end of the trajectory. If shifting
        backward, extends from the beginning with reversed velocities.

        Args:
            lmp: Active LAMMPS instance.
            pt: Shift point (frame index, 0-indexed).
            T: Temperature string.
            s: s-field value string.
            g: g-field value string.

        Returns:
            Tuple from :meth:`_accept_or_reject`.
        """
        cfg = self.config
        log = initialize_log_dict(cfg.nloops, cfg.n_particles)

        if pt < cfg.nloops / 2:  # Forward
            pos_start = self.TPS_pos[-1]
            vel_start = self.TPS_vel[-1]
            log["pos"][: cfg.nloops - pt + 1] = self.TPS_pos[pt:]
            log["vel"][: cfg.nloops - pt + 1] = self.TPS_vel[pt:]
            log["pe"][: cfg.nloops - pt + 1]  = self.TPS_PE[pt:]
            log["ke"][: cfg.nloops - pt + 1]  = self.TPS_KE[pt:]
        else:  # Backward
            pos_start = self.TPS_pos[0]
            vel_start = -self.TPS_vel[0]
            log["pos"][cfg.nloops - pt:] = self.TPS_pos[: pt + 1]
            log["vel"][cfg.nloops - pt:] = self.TPS_vel[: pt + 1]
            log["pe"][cfg.nloops - pt:]  = self.TPS_PE[: pt + 1]
            log["ke"][cfg.nloops - pt:]  = self.TPS_KE[: pt + 1]

        lmp.command("reset_timestep 0")
        self.thermostat.fix(lmp, T)
        lmp.command(f"thermo {cfg.nstout}")
        lmp.command("thermo_style custom step temp ke pe etotal press")
        lmp.command("thermo_modify norm no")
        lmp.scatter_atoms("x", 1, 3, npy2Cdouble(pos_start.flatten()))
        lmp.scatter_atoms("v", 1, 3, npy2Cdouble(vel_start.flatten()))

        if pt < cfg.nloops / 2:  # Forward
            for i in range(pt):
                lmp.command(f"run {cfg.nstout}")
                log["pos"][cfg.nloops - pt + i + 1] = np.array(lmp.gather_atoms("x", 1, 3)).reshape(-1, 3)
                log["vel"][cfg.nloops - pt + i + 1] = np.array(lmp.gather_atoms("v", 1, 3)).reshape(-1, 3)
                log["pe"][cfg.nloops - pt + i + 1]  = lmp.get_thermo("pe")
                log["ke"][cfg.nloops - pt + i + 1]  = lmp.get_thermo("ke")
        else:  # Backward
            for i in range(cfg.nloops - pt):
                lmp.command(f"run {cfg.nstout}")
                log["pos"][cfg.nloops - pt - i - 1] = np.array(lmp.gather_atoms("x", 1, 3)).reshape(-1, 3)
                log["vel"][cfg.nloops - pt - i - 1] = np.array(lmp.gather_atoms("v", 1, 3)).reshape(-1, 3)
                log["pe"][cfg.nloops - pt - i - 1]  = lmp.get_thermo("pe")
                log["ke"][cfg.nloops - pt - i - 1]  = lmp.get_thermo("ke")

        self.thermostat.unfix(lmp)

        return self._accept_or_reject(log, s, g)

    # ── Metropolis-Hastings acceptance ────────────────────────────────────

    def _accept_or_reject(self, log: dict, s: str, g: str) -> tuple:
        """Apply the Metropolis-Hastings criterion and update trajectory state.

        Computes the acceptance factor ``exp(-s*dK - g*dE)`` where ``dK`` and
        ``dE`` are the changes in activity and total energy. Accepts the
        proposed trajectory with probability ``min(1, factor)`` and updates
        the stored trajectory state in-place on acceptance.

        Args:
            log: Proposed trajectory dict with keys ``"pos"``, ``"vel"``,
                ``"pe"``, ``"ke"`` (as produced by a move method).
            s: s-field value string.
            g: g-field value string.

        Returns:
            Tuple ``(r, cur_E, cur_K, new_E, new_K, dE, dK, factor, accepted)``
            where ``r`` is the uniform random variate, energies are summed over
            the trajectory, ``factor`` is the MH weight, and ``accepted`` is
            ``1`` if accepted or ``0`` if rejected.
        """
        cur_E = float(np.sum(self.TPS_PE + self.TPS_KE))
        new_E = float(np.sum(log["pe"] + log["ke"]))
        dE = new_E - cur_E

        cur_K = self.TPS_K
        new_K = compute_activity(log["pos"], self.config.box_size)
        dK = new_K - cur_K

        r = np.random.uniform()
        factor = np.exp(-float(s) * dK - float(g) * dE)

        if r < factor:
            self.TPS_pos = log["pos"]
            self.TPS_vel = log["vel"]
            self.TPS_PE  = log["pe"]
            self.TPS_KE  = log["ke"]
            self.TPS_K   = new_K
            accepted = 1
        else:
            accepted = 0

        return r, cur_E, cur_K, new_E, new_K, dE, dK, factor, accepted
