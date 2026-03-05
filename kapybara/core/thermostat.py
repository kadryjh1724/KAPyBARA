"""LAMMPS thermostat strategy classes.

Implements the Strategy pattern for thermostat management. Each concrete
subclass applies the appropriate LAMMPS ``fix`` commands for a specific
thermostat type and removes them cleanly in ``unfix``.
"""

from abc import ABC, abstractmethod

import numpy as np


class Thermostat(ABC):
    """Abstract base class for LAMMPS thermostat strategies."""

    @abstractmethod
    def fix(self, lmp, T: str) -> None:
        """Apply thermostat fixes to a LAMMPS instance.

        Args:
            lmp: Active LAMMPS instance.
            T: Temperature as a formatted string (e.g. ``"0.4500"``).
        """

    @abstractmethod
    def unfix(self, lmp) -> None:
        """Remove thermostat fixes from a LAMMPS instance.

        Args:
            lmp: Active LAMMPS instance.
        """


class NoseHooverThermostat(Thermostat):
    """Nosé-Hoover NVT thermostat (LAMMPS ``fix nvt``).

    Uses a damping time of 1.5 (in reduced LJ units) and resets linear
    momentum every step.
    """

    def fix(self, lmp, T: str) -> None:
        """Apply NVT and momentum-rescaling fixes.

        Args:
            lmp: Active LAMMPS instance.
            T: Target temperature string.
        """
        lmp.command(f"fix 1 all nvt temp {T} {T} 1.5")
        lmp.command(f"fix 2 all momentum 1 linear 1 1 1 rescale")

    def unfix(self, lmp) -> None:
        """Remove momentum and NVT fixes.

        Args:
            lmp: Active LAMMPS instance.
        """
        lmp.command("unfix 2")
        lmp.command("unfix 1")


class LangevinThermostat(Thermostat):
    """Langevin thermostat (LAMMPS ``fix langevin`` + ``fix nve``).

    Attributes:
        gamma: Damping coefficient in reduced LJ units.
    """

    def __init__(self, gamma: float):
        """Initialise with a damping coefficient.

        Args:
            gamma: Langevin damping coefficient (positive float).
        """
        self.gamma = gamma

    def fix(self, lmp, T: str) -> None:
        """Apply Langevin, NVE, and momentum-rescaling fixes.

        A random seed is drawn from ``numpy`` for the Langevin noise.

        Args:
            lmp: Active LAMMPS instance.
            T: Target temperature string.
        """
        r = np.random.randint(1e6)
        lmp.command(f"fix 1 all langevin {T} {T} {self.gamma} {r} zero yes")
        lmp.command(f"fix 2 all nve")
        lmp.command(f"fix 3 all momentum 1 linear 1 1 1 rescale")

    def unfix(self, lmp) -> None:
        """Remove momentum, NVE, and Langevin fixes.

        Args:
            lmp: Active LAMMPS instance.
        """
        lmp.command("unfix 3")
        lmp.command("unfix 2")
        lmp.command("unfix 1")


class MSCThermostat(Thermostat):
    """Momentum-space velocity-rescaling thermostat (LAMMPS ``fix temp/rescale``).

    Attributes:
        nstout: Rescaling frequency (every ``nstout`` steps).
    """

    def __init__(self, nstout: int):
        """Initialise with the output/rescaling frequency.

        Args:
            nstout: Number of MD steps between rescaling events.
        """
        self.nstout = nstout

    def fix(self, lmp, T: str) -> None:
        """Apply velocity-rescaling, NVE, and momentum fixes.

        Args:
            lmp: Active LAMMPS instance.
            T: Target temperature string.
        """
        lmp.command(f"fix 1 all temp/rescale {self.nstout} {T} {T} 0.01 1.0")
        lmp.command(f"fix 2 all nve")
        lmp.command(f"fix 3 all momentum 1 linear 1 1 1 rescale")

    def unfix(self, lmp) -> None:
        """Remove momentum, NVE, and velocity-rescaling fixes.

        Args:
            lmp: Active LAMMPS instance.
        """
        lmp.command("unfix 3")
        lmp.command("unfix 2")
        lmp.command("unfix 1")


def create_thermostat(config) -> Thermostat:
    """Factory: instantiate the correct Thermostat from SimulationConfig.

    Args:
        config: :class:`~kapybara.config.schema.SimulationConfig` instance.

    Returns:
        A concrete :class:`Thermostat` subclass instance.

    Raises:
        ValueError: If ``config.thermostat`` is not a recognised type.
    """
    if config.thermostat == "Nose-Hoover":
        return NoseHooverThermostat()
    elif config.thermostat == "Langevin":
        return LangevinThermostat(config.gamma)
    elif config.thermostat == "MSC":
        return MSCThermostat(config.nstout)
    else:
        raise ValueError(f"Unknown thermostat type: {config.thermostat}")
