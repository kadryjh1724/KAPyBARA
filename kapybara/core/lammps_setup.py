"""LAMMPS instance creation and Kob-Andersen system setup.

Provides factory functions for creating LAMMPS instances (with suppressed
output) and for configuring the binary Lennard-Jones Kob-Andersen
glass-forming mixture.
"""

import numpy as np
from lammps import lammps

from kapybara.config.schema import SimulationConfig


def create_lammps_instance() -> lammps:
    """Create a LAMMPS MPI instance with all output suppressed.

    Returns:
        A new :class:`lammps.lammps` instance ready for use.
    """
    return lammps(name="mpi", cmdargs=["-screen", "none", "-log", "none"])


def setup_kob_andersen(lmp: lammps, config: SimulationConfig) -> None:
    """Configure a Kob-Andersen binary LJ mixture in an existing LAMMPS instance.

    Sets up units, atom style, pair style (LJ/cut, cutoff 2.5), periodic
    boundary conditions, a cubic simulation box, and randomly places
    ``N_A`` type-A and ``N_B`` type-B particles. Sets masses and the
    standard Kob-Andersen pair coefficients (AA, BB, AB).

    Args:
        lmp: An active LAMMPS instance (typically from
            :func:`create_lammps_instance`).
        config: Frozen :class:`~kapybara.config.schema.SimulationConfig`
            providing ``box_size``, ``N_A``, and ``N_B``.
    """
    r = np.random.randint(1e6, size=3)

    lmp.command("units lj")
    lmp.command("atom_style atomic")
    lmp.command("atom_modify map yes")
    lmp.command("pair_style lj/cut 2.5")
    lmp.command("boundary p p p")

    L = config.box_size
    lmp.command(f"region box block 0 {L} 0 {L} 0 {L}")
    lmp.command("create_box 2 box")
    lmp.command(f"create_atoms 1 random {config.N_A} {r[0]} box")
    lmp.command(f"create_atoms 2 random {config.N_B} {r[1]} box")

    lmp.command("mass 1 1.0")
    lmp.command("mass 2 1.0")
    lmp.command("pair_coeff 1 1 1.0 1.0")
    lmp.command("pair_coeff 2 2 0.5 0.88")
    lmp.command("pair_coeff 1 2 1.5 0.8")
    lmp.command("group typeA type 1")
    lmp.command("group typeB type 2")
