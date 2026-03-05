"""Shared TPS runner base class for single-field (Tg and Ts) simulations.

``_RunnerBase`` implements the complete relax + acquisition loop, parameterised
by ``field_axis`` (``"g"`` or ``"s"``). Subclasses set ``field_axis`` as a
class attribute; ``run()`` assigns ``field_value`` to the active field and reads
the fixed field scalar from ``config``.
"""

import os
import time
import numpy as np

from kapybara.config.schema import SimulationConfig
from kapybara.config.paths import PathManager
from kapybara.state.db import StateDB
from kapybara.sampling.moves import TPSMoves
from kapybara.core.lammps_setup import create_lammps_instance, setup_kob_andersen
from kapybara.core.activity import compute_activity
from kapybara.utils.trim import trim_csv


class _RunnerBase:
    """Shared TPS runner logic for single-field (Tg or Ts) simulations.

    Subclasses must set the class attribute ``field_axis`` to either ``"g"`` or
    ``"s"`` to determine which field variable receives ``field_value`` and which
    is held fixed at the config-provided scalar (``config.s[0]`` or
    ``config.g[0]`` respectively).

    Attributes:
        field_axis: ``"g"`` for T-g runs, ``"s"`` for T-s runs. Must be set by
            the subclass.
        config: Frozen simulation configuration.
        paths: Path manager for all simulation directories.
        state_db: Central SQLite state tracker (queue-mode or direct).
        moves: TPS moves instance shared by the worker.
    """

    field_axis: str  # must be "g" or "s" — set by subclass

    def __init__(self, config: SimulationConfig, paths: PathManager,
                 state_db: StateDB, moves: TPSMoves):
        """Initialise the runner with simulation context.

        Args:
            config: Frozen :class:`~kapybara.config.schema.SimulationConfig`.
            paths: :class:`~kapybara.config.paths.PathManager` instance.
            state_db: :class:`~kapybara.state.db.StateDB` instance (queue-mode).
            moves: :class:`~kapybara.sampling.moves.TPSMoves` instance.
        """
        self.config = config
        self.paths = paths
        self.state_db = state_db
        self.moves = moves

    # ── Public entry point ────────────────────────────────────────────────

    def run(self, T: str, field_value: str, replica_index: int) -> None:
        """Execute relaxation + acquisition TPS for one (T, field_value, replica).

        Handles all restart scenarios by scanning existing ``.npy`` checkpoint
        files. Loads the initial trajectory from the prerun output (root field)
        or from the parent field's n_branch checkpoint (non-root). Reports
        progress to StateDB and marks the replica completed on success.

        Args:
            T: Temperature string.
            field_value: Active field value string (g or s, per ``field_axis``).
            replica_index: Zero-based replica index.

        Raises:
            Exception: Re-raises any error after recording ``'failed'`` status
                in StateDB and closing the LAMMPS instance.
        """
        cfg   = self.config
        paths = self.paths

        # Assign active and fixed field values
        if self.field_axis == "g":
            g = field_value
            s = cfg.s[0]
        else:  # "s"
            s = field_value
            g = cfg.g[0]

        # field_value is always used as the path subdirectory (active field)
        trj_rel  = os.path.join(paths.step2_trj, T, field_value,
                                f"relax_{replica_index:02d}")
        trj_acq  = os.path.join(paths.step2_trj, T, field_value,
                                f"acqui_{replica_index:02d}")
        ene_rel  = os.path.join(paths.step2_ene, T, field_value,
                                f"relax_{replica_index:02d}")
        ene_acq  = os.path.join(paths.step2_ene, T, field_value,
                                f"acqui_{replica_index:02d}")
        csv_path = os.path.join(paths.step2_csv, T, field_value,
                                f"{replica_index:02d}.csv")

        tag = f"T={T} {self.field_axis}={field_value} r#{replica_index:02d}"

        # ── Restart detection ─────────────────────────────────────────────
        restart_idx      = 0
        start_from_relax = True
        continue_relax   = False
        relax_idx        = 0
        acqui_idx        = 0

        # Case I: look for existing acqui dumps (newest first)
        for idx in reversed(cfg.dump_acqui):
            if os.path.exists(f"{trj_acq}_{idx}.npy"):
                restart_idx = idx
                start_from_relax = False
                break

        # Case II: if no acqui dump, look for relax dumps
        if start_from_relax:
            for idx in reversed(cfg.dump_relax):
                if os.path.exists(f"{trj_rel}_{idx}.npy"):
                    if idx == cfg.dump_relax[-1]:
                        # Last relax dump exists → all relax done, start acqui
                        restart_idx = 0
                        start_from_relax = False
                    else:
                        restart_idx = idx
                        continue_relax = True
                    break

        # ── Load initial trajectory ────────────────────────────────────────
        if not start_from_relax:
            if restart_idx != 0:
                # Resume acqui from mid-run
                try:
                    pro_trj = np.load(f"{trj_acq}_{restart_idx}.npy")
                    pro_ene = np.load(f"{ene_acq}_{restart_idx}.npy")
                except Exception as e:
                    print(f"{type(self).__name__}: failed to load acqui restart "
                          f"{tag}: {e}.")
                    raise
                trim_csv(csv_path, restart_idx, "ACQUI")
                acqui_idx = restart_idx + 1
            else:
                # All relax done; start acqui from the last relax dump
                last_relax = cfg.dump_relax[-1]
                try:
                    pro_trj = np.load(f"{trj_rel}_{last_relax}.npy")
                    pro_ene = np.load(f"{ene_rel}_{last_relax}.npy")
                except Exception as e:
                    print(f"{type(self).__name__}: failed to load relax→acqui "
                          f"restart {tag}: {e}.")
                    raise
                trim_csv(csv_path, last_relax, "RELAX")

        elif continue_relax:
            # Resume relax from mid-run
            try:
                pro_trj = np.load(f"{trj_rel}_{restart_idx}.npy")
                pro_ene = np.load(f"{ene_rel}_{restart_idx}.npy")
            except Exception as e:
                print(f"{type(self).__name__}: failed to load relax restart "
                      f"{tag}: {e}.")
                raise
            trim_csv(csv_path, restart_idx, "RELAX")
            relax_idx = restart_idx + 1

        else:
            # Case III: fresh start — load from prerun or parent field
            dep = self.state_db.get_dependency(T, field_value)

            if dep is None:
                # Root field: load from step1 prerun output
                try:
                    pro_trj = np.load(
                        os.path.join(paths.step1_trj, T,
                                     f"{replica_index:02d}.npy"))
                except Exception as e:
                    print(f"{type(self).__name__}: no prerun trajectory "
                          f"T={T} r#{replica_index}: {e}.")
                    raise
                try:
                    pro_ene = np.load(
                        os.path.join(paths.step1_ene, T,
                                     f"{replica_index:02d}.npy"))
                except Exception as e:
                    print(f"{type(self).__name__}: no prerun energy "
                          f"T={T} r#{replica_index}: {e}.")
                    raise
            else:
                # Non-root: branch from parent field's n_branch-1 dump
                branch_idx = cfg.n_branch - 1
                try:
                    pro_trj = np.load(
                        os.path.join(paths.step2_trj, T, dep,
                                     f"acqui_{replica_index:02d}_{branch_idx}.npy"))
                except Exception as e:
                    print(f"{type(self).__name__}: no parent trajectory "
                          f"T={T} {self.field_axis}_dep={dep} "
                          f"r#{replica_index}: {e}.")
                    raise
                try:
                    pro_ene = np.load(
                        os.path.join(paths.step2_ene, T, dep,
                                     f"acqui_{replica_index:02d}_{branch_idx}.npy"))
                except Exception as e:
                    print(f"{type(self).__name__}: no parent energy "
                          f"T={T} {self.field_axis}_dep={dep} "
                          f"r#{replica_index}: {e}.")
                    raise

            # Write CSV header for fresh run
            with open(csv_path, "w") as f:
                f.write(
                    "RUN_TYPE,SAMPLING_TYPE,RUN_IDX,TPS_PNT,RND_NUM,"
                    "CUR_E,CUR_K,NEW_E,NEW_K,\u0394E,\u0394K,"
                    "exp(-s\u0394K-g\u0394E),STATUS,RUN_TIME\n"
                )

        # Initialise TPSMoves trajectory state
        self.moves.TPS_pos = pro_trj[:, :, :3]
        self.moves.TPS_vel = pro_trj[:, :, 3:]
        self.moves.TPS_PE  = pro_ene[0]
        self.moves.TPS_KE  = pro_ene[1]
        self.moves.TPS_K   = compute_activity(self.moves.TPS_pos, cfg.box_size)

        # Pre-generate random sequences
        pts_relax = np.random.randint(low=1, high=cfg.nloops - 1, size=cfg.n_relax)
        pts_acqui = np.random.randint(low=1, high=cfg.nloops - 1, size=cfg.n_acqui)
        run_relax = np.random.uniform(size=cfg.n_relax)
        run_acqui = np.random.uniform(size=cfg.n_acqui)

        # ── LAMMPS setup ──────────────────────────────────────────────────
        lmp = None
        try:
            lmp = create_lammps_instance()
            setup_kob_andersen(lmp, cfg)
            lmp.command("neigh_modify every 1 delay 5 check yes")

            self.state_db.update_tps_state(T, field_value, replica_index,
                                           "relax", 0, "running")

            # ── Relaxation phase ──────────────────────────────────────────
            time_relax = []
            if start_from_relax:
                for raw_i, (pt, p) in enumerate(
                        zip(pts_relax[relax_idx:], run_relax[relax_idx:])):
                    idx = raw_i + relax_idx

                    stype, result, elapsed = self._move(lmp, pt, p, T, s, g)
                    self._write_csv(csv_path, "RELAX", idx, pt, stype,
                                    result, elapsed)
                    time_relax.append(elapsed)

                    if idx in cfg.dump_relax:
                        np.save(f"{trj_rel}_{idx}",
                                np.concatenate((self.moves.TPS_pos,
                                                self.moves.TPS_vel), axis=2))
                        np.save(f"{ene_rel}_{idx}",
                                np.vstack((self.moves.TPS_PE, self.moves.TPS_KE)))
                        self.state_db.update_tps_state(
                            T, field_value, replica_index,
                            "relax", idx + 1, "running")

                print(f"TPS(relax): {tag} done in {sum(time_relax):.1f}s.")

            # ── Acquisition phase ─────────────────────────────────────────
            time_acqui = []
            for raw_i, (pt, p) in enumerate(
                    zip(pts_acqui[acqui_idx:], run_acqui[acqui_idx:])):
                idx = raw_i + acqui_idx

                stype, result, elapsed = self._move(lmp, pt, p, T, s, g)
                self._write_csv(csv_path, "ACQUI", idx, pt, stype,
                                result, elapsed)
                time_acqui.append(elapsed)

                if idx in cfg.dump_acqui:
                    np.save(f"{trj_acq}_{idx}",
                            np.concatenate((self.moves.TPS_pos,
                                            self.moves.TPS_vel), axis=2))
                    np.save(f"{ene_acq}_{idx}",
                            np.vstack((self.moves.TPS_PE, self.moves.TPS_KE)))
                    self.state_db.update_tps_state(
                        T, field_value, replica_index,
                        "acqui", idx + 1, "running")

            print(f"TPS(acqui): {tag} done in {sum(time_acqui):.1f}s.")

            self.state_db.mark_tps_completed(T, field_value, replica_index)

        except Exception as e:
            print(f"{type(self).__name__}: error {tag}: {e}.")
            self.state_db.update_tps_state(T, field_value, replica_index,
                                           "acqui", 0, "failed", error=str(e))
            raise
        finally:
            if lmp is not None:
                lmp.close()

    # ── Helpers ───────────────────────────────────────────────────────────

    def _move(self, lmp, pt: int, p: float,
              T: str, s: str, g: str) -> tuple:
        """Execute one TPS move (shoot or shift) and measure wall time.

        Dispatches to :attr:`~kapybara.sampling.moves.TPSMoves.shoot` or
        :attr:`~kapybara.sampling.moves.TPSMoves.shift` based on the random
        variate ``p`` compared to ``config.p_shoot``.

        Args:
            lmp: Active LAMMPS instance.
            pt: Shooting/shifting point (frame index).
            p: Uniform random variate in [0, 1) drawn before calling.
            T: Temperature string.
            s: s-field value string.
            g: g-field value string.

        Returns:
            Tuple ``(stype, result, elapsed_seconds)`` where ``stype`` is
            ``"SHOOT"`` or ``"SHIFT"``, ``result`` is the tuple returned by
            the move method, and ``elapsed_seconds`` is the wall time.
        """
        t0 = time.perf_counter()
        if p < self.config.p_shoot:
            stype  = "SHOOT"
            result = self.moves.shoot(lmp, pt, T, s, g)
        else:
            stype  = "SHIFT"
            result = self.moves.shift(lmp, pt, T, s, g)
        return stype, result, time.perf_counter() - t0

    @staticmethod
    def _write_csv(csv_path: str, run_type: str, idx: int, pt: int,
                   stype: str, result: tuple, elapsed: float) -> None:
        """Append one completed-move record to the per-replica CSV log.

        Each line records move type, phase, index, shooting/shifting point,
        random variate, energies, activity, MH factor, acceptance, and timing.

        Args:
            csv_path: Path to the per-replica CSV file.
            run_type: ``"RELAX"`` or ``"ACQUI"``.
            idx: Zero-based run index within the phase.
            pt: Shooting or shifting point (frame index).
            stype: ``"SHOOT"`` or ``"SHIFT"``.
            result: Tuple from
                :meth:`~kapybara.sampling.moves.TPSMoves._accept_or_reject`.
            elapsed: Wall-clock time for this move in seconds.
        """
        r, cur_E, cur_K, new_E, new_K, dE, dK, factor, accepted = result
        with open(csv_path, "a") as f:
            f.write(
                f"{run_type},{stype},{idx},{pt},{r:.4f},"
                f"{cur_E:.5E},{cur_K:.5E},{new_E:.5E},{new_K:.5E},"
                f"{dE:.5E},{dK:.5E},{factor:.3E},{accepted},{elapsed:.2f}\n"
            )
