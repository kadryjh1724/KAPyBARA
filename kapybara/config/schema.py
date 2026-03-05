"""Immutable simulation configuration dataclass.

SimulationConfig is the single source of truth for all simulation parameters.
It is constructed by :func:`kapybara.config.loader.load_config` and passed by
reference throughout the package. All fields are frozen at construction.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class SimulationConfig:
    """Immutable simulation configuration parsed from YAML.

    All fields are set at construction and cannot be modified. Derived fields
    (``nsteps``, ``nloops``, ``dump_relax``, ``dump_acqui``, etc.) are computed
    by :func:`kapybara.config.loader._parse` and stored alongside the raw YAML
    values.

    Attributes:
        job_name: Unique name for the simulation run.
        work_directory: Root directory for all output files.
        partition: SLURM partition to submit jobs to.
        exclude: List of node names to exclude from SLURM allocation.
        n_particles: Total number of particles (A + B).
        N_A: Number of type-A particles (= ``4 * n_particles // 5``).
        N_B: Number of type-B particles (= ``n_particles - N_A``).
        density: Number density of the system.
        box_size: Cubic box side length (= ``cbrt(n_particles / density)``).
        n_replica: Number of independent TPS replicas per (T, field) point.
        runtype: Simulation mode — ``"g"``, ``"s"``, or ``"sg"``.
        T: Formatted temperature strings (e.g. ``["0.4500", "0.5000"]``).
        s: Formatted s-field value strings.
        g: Formatted g-field value strings.
        n_decimals: Decimal places for [T, s, g] string formatting.
        dt: MD time step.
        nstout: Number of MD steps between output frames.
        t_obs: Observation time window length.
        nsteps: Total MD steps per trajectory (= ``t_obs / dt``).
        nloops: Number of output frames per trajectory (= ``nsteps / nstout``).
        t_equil: Equilibration time for prerun.
        nsteps_equil: MD steps for equilibration (= ``t_equil / dt``).
        nloops_equil: Output frames for equilibration.
        thermostat: Thermostat type — ``"Nose-Hoover"``, ``"Langevin"``, or
            ``"MSC"``.
        gamma: Langevin damping coefficient; only used for Langevin thermostat.
        p_shoot: Probability of attempting a shooting move.
        p_shift: Probability of attempting a shifting move.
        one_way_shoot: If ``True``, use one-way (forward-only) shooting.
        one_way_shift: If ``True``, use one-way (forward-only) shifting.
        n_relax: Number of relaxation TPS moves per replica.
        n_acqui: Number of acquisition TPS moves per replica.
        n_branch: Acquisition run index at which child jobs may branch from a
            running parent.
        n_dump: ``[relax_dump_count, acqui_dump_count]``.
        dump_relax: List of relax run indices at which trajectory dumps are
            written.
        dump_acqui: List of acqui run indices at which trajectory dumps are
            written.
    """

    # Job identification
    job_name: str
    work_directory: str
    partition: str
    exclude: List[str]

    # Particle system
    n_particles: int
    N_A: int
    N_B: int
    density: float
    box_size: float
    n_replica: int

    # Run mode
    runtype: str                # "s", "g", "sg"

    # Field parameters (formatted string lists)
    T: List[str]
    s: List[str]
    g: List[str]
    n_decimals: List[int]       # [T_decimals, s_decimals, g_decimals]

    # MD parameters
    dt: float
    nstout: int
    t_obs: float
    nsteps: int                 # derived: t_obs / dt
    nloops: int                 # derived: nsteps / nstout
    t_equil: float
    nsteps_equil: int           # derived: t_equil / dt
    nloops_equil: int           # derived: nsteps_equil / nstout

    # Thermostat
    thermostat: str             # "Langevin", "Nose-Hoover", "MSC"
    gamma: Optional[float]      # required only for Langevin

    # TPS parameters
    p_shoot: float
    p_shift: float
    one_way_shoot: bool
    one_way_shift: bool

    # Run counts & branching
    n_relax: int
    n_acqui: int
    n_branch: int
    n_dump: List[int]           # [relax_dump_count, acqui_dump_count]
    dump_relax: List[int]       # derived: dump index list
    dump_acqui: List[int]       # derived: dump index list
    