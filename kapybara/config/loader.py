"""YAML configuration loader and validator.

Parses a YAML config file, validates all parameters, and returns an immutable
:class:`~kapybara.config.schema.SimulationConfig`. The internal helper
:func:`_parse` performs all validation and raises
:class:`~kapybara.utils.errors._ValidationError` on invalid values.
"""

import os
import yaml
import numpy as np
from typing import List, Union
from rich.console import Console

from kapybara.config.schema import SimulationConfig
from kapybara.utils.errors import _ValidationError
from kapybara.utils.cstring import prettyWarning, prettyError, prettyNotification
from kapybara.utils.convert import npy2str


def load_config(config_path: str, quiet: bool = False) -> SimulationConfig:
    """Load YAML config, validate, and return a frozen SimulationConfig.

    Raises:
        FileNotFoundError: if config file does not exist.
        _ValidationError: if any config parameter is invalid.
    """
    console = Console()
    console_stderr = Console(stderr=True)

    config_path = os.path.realpath(config_path)

    try:
        with open(config_path, "r") as f:
            raw = yaml.safe_load(f)
    except FileNotFoundError as e:
        prettyError(console_stderr, e,
            "Configuration file not found in the given PATH. Check again.",
            config_path)
        raise

    try:
        config = _parse(raw, console, console_stderr, quiet)
    except _ValidationError as e:
        prettyError(console_stderr, e.exc, e.message1, e.message2)
        raise

    if not quiet:
        prettyNotification(console,
            f"Simulation {config.job_name} configuration file parsed successfully.")

    return config


def _parse(raw: dict, console: Console, console_stderr: Console,
           quiet: bool) -> SimulationConfig:
    """Parse and validate a raw YAML dictionary into a SimulationConfig.

    Validates every parameter, emits Rich-formatted warnings for non-fatal
    issues, and raises :class:`~kapybara.utils.errors._ValidationError` for
    any value that fails validation. Derived fields (``nsteps``, ``dump_relax``,
    etc.) are computed and included in the returned config.

    Args:
        raw: Dictionary loaded directly from YAML.
        console: Rich Console for stdout notifications.
        console_stderr: Rich Console for stderr warnings and errors.
        quiet: If ``True``, suppress informational output.

    Returns:
        A fully validated, frozen :class:`~kapybara.config.schema.SimulationConfig`.

    Raises:
        _ValidationError: If any parameter fails validation.
    """

    n_warn = 0

    # ── Job identification ───────────────────────────────────────────────

    job_name        = raw["job_name"]
    work_directory  = raw["work_directory"]
    partition       = raw["partition"]
    exclude         = raw.get("exclude", [])

    # ── Particle system ──────────────────────────────────────────────────

    n_particles = raw["n_particles"]
    if not isinstance(n_particles, int) or n_particles <= 0:
        raise _ValidationError(ValueError(),
            "'n_particles' must be a positive integer.",
            f"current value is {n_particles}.")

    N_A = 4 * n_particles // 5
    N_B = n_particles - N_A

    if n_particles % 5 != 0:
        prettyWarning(console_stderr,
            "'n_particles' should be divisible by 5: this would not cause error, "
            "but A:B particle ratio would not be exactly 4:1 in this case.",
            f"current value is {n_particles} with N_A : N_B = {N_A} : {N_B}.")
        n_warn += 1

    density = raw["density"]
    if density <= 0:
        raise _ValidationError(ValueError(),
            "'density' should be a positive real number.",
            f"current value is {density}.")
    box_size = np.round(np.cbrt(n_particles / density), 4)

    n_replica = raw["n_replica"]
    if not isinstance(n_replica, int) or n_replica <= 0:
        raise _ValidationError(ValueError(),
            "'n_replica' must be a positive integer.",
            f"current value is {n_replica}.")

    # ── MD parameters ────────────────────────────────────────────────────

    dt = raw["dt"]
    if dt <= 0:
        raise _ValidationError(ValueError(),
            "'dt' must be a positive real number.",
            f"current value is {dt}.")

    nstout = raw["nstout"]
    if not isinstance(nstout, int) or nstout <= 0:
        raise _ValidationError(ValueError(),
            "'nstout' must be a positive integer.",
            f"current value is {nstout}.")

    t_obs = raw["t_obs"]
    if t_obs <= 0:
        raise _ValidationError(ValueError(),
            "'t_obs' must be a positive real number.",
            f"current value is {t_obs}.")

    t_equil = raw["t_equil"]
    if t_equil <= 0:
        raise _ValidationError(ValueError(),
            "'t_equil' must be a positive real number.",
            f"current value is {t_equil}.")

    nsteps       = int(t_obs / dt)
    nloops       = int(nsteps / nstout)
    nsteps_equil = int(t_equil / dt)
    nloops_equil = int(nsteps_equil / nstout)

    # ── Thermostat ───────────────────────────────────────────────────────

    thermostat = raw["thermostat"]
    if thermostat not in ("Nose-Hoover", "Langevin", "MSC"):
        raise _ValidationError(ValueError(),
            "Thermostat type should be 'Nose-Hoover', 'Langevin' or 'MSC'.",
            f"current thermostat is {thermostat}.")

    gamma = None
    if thermostat == "Langevin":
        gamma = raw["gamma"]
        if gamma <= 0:
            raise _ValidationError(ValueError(),
                "'gamma' must be a positive real number.",
                f"current value is {gamma}.")

    # ── Run type ─────────────────────────────────────────────────────────

    runtype = raw["runtype"]
    if runtype not in ("s", "g", "sg"):
        raise _ValidationError(ValueError(),
            "Run type should be 's', 'g' or 'sg'.",
            f"current run type is {runtype}.")

    # ── Field parameters ─────────────────────────────────────────────────

    T_npy = _expand_parameter(raw["T"])
    s_npy = _expand_parameter(raw["s"])
    g_npy = _expand_parameter(raw["g"])

    if runtype == "s" and g_npy.shape != ():
        raise _ValidationError(ValueError(),
            "For run type 's', g-field value should be a single real number.",
            f"current g field values are {g_npy}.")
    if runtype == "g" and s_npy.shape != ():
        raise _ValidationError(ValueError(),
            "For run type 'g', s-field value should be a single real number.",
            f"current s field values are {s_npy}.")

    n_decimals = raw["n_decimals"]
    for n in n_decimals:
        if not isinstance(n, int) or n <= 0:
            raise _ValidationError(ValueError(),
                "'n_decimals' must be a list of positive integers.",
                f"current value is {n_decimals}.")

    T = npy2str(T_npy, n_decimals[0])
    s = npy2str(s_npy, n_decimals[1])
    g = npy2str(g_npy, n_decimals[2])

    # ── TPS parameters ───────────────────────────────────────────────────

    one_way_shoot = raw["one_way_shoot"]
    one_way_shift = raw["one_way_shift"]

    p_shoot = raw["p_shoot"]
    if not 0 <= p_shoot <= 1:
        raise _ValidationError(ValueError(),
            "'p_shoot' must be a real number between 0 and 1.",
            f"current value is {p_shoot}.")

    p_shift = raw["p_shift"]
    if not 0 <= p_shift <= 1:
        raise _ValidationError(ValueError(),
            "'p_shift' must be a real number between 0 and 1.",
            f"current value is {p_shift}.")

    if abs(p_shoot + p_shift - 1.0) > 1e-10:
        raise _ValidationError(ValueError(),
            "'p_shoot' and 'p_shift' must add up to 1.0.",
            f"current values are p_shoot = {p_shoot}, p_shift = {p_shift}.")

    # ── Run counts & branching ───────────────────────────────────────────

    n_relax = raw["n_relax"]
    if not isinstance(n_relax, int) or n_relax <= 0:
        raise _ValidationError(ValueError(),
            "'n_relax' must be a positive integer.",
            f"current value is {n_relax}.")

    n_acqui = raw["n_acqui"]
    if not isinstance(n_acqui, int) or n_acqui <= 0:
        raise _ValidationError(ValueError(),
            "'n_acqui' must be a positive integer.",
            f"current value is {n_acqui}.")

    n_branch = raw["n_branch"]
    if not isinstance(n_branch, int) or n_branch <= 0 or n_branch > n_acqui:
        raise _ValidationError(ValueError(),
            "'n_branch' must be a positive integer, smaller than or equal to 'n_acqui'.",
            f"current value is {n_branch}.")

    n_dump = raw["n_dump"]
    if n_relax % n_dump[0] != 0:
        raise _ValidationError(ValueError(),
            "Last relaxation run is not dumped with current 'n_dump'.",
            f"current n_relax = {n_relax}, current n_dump for relaxation run = {n_dump[0]}.")
    if n_acqui % n_dump[1] != 0:
        raise _ValidationError(ValueError(),
            "Last acquisition run is not dumped with current 'n_dump'.",
            f"current n_acqui = {n_acqui}, current n_dump for acquisition run = {n_dump[1]}.")

    dump_relax = list(map(int, np.linspace(0, n_relax, n_dump[0] + 1)[1:] - 1))
    dump_acqui = list(map(int, np.linspace(0, n_acqui, n_dump[1] + 1)[1:] - 1))

    if n_branch - 1 not in dump_acqui:
        idx = dump_acqui[np.searchsorted(dump_acqui, n_branch - 1)]
        prettyWarning(console_stderr,
            "'n_branch' index is not included in acquisition run dump indices. "
            "This would not cause error, but you should be aware of this behavior.",
            f"KAPyBARA will automatically use {idx}'th trajectory for the next run, "
            f"not {n_branch - 1}'th trajectory.")
        n_warn += 1

    # ── Build config ─────────────────────────────────────────────────────

    if not quiet and n_warn > 0:
        prettyNotification(console,
            f"{n_warn} warning(s) were generated during parsing. "
            "Please check before proceeding.")

    return SimulationConfig(
        job_name        = job_name,
        work_directory  = work_directory,
        partition       = partition,
        exclude         = exclude,
        n_particles     = n_particles,
        N_A             = N_A,
        N_B             = N_B,
        density         = density,
        box_size        = float(box_size),
        runtype         = runtype,
        T               = T,
        s               = s,
        g               = g,
        n_decimals      = n_decimals,
        t_obs           = t_obs,
        dt              = dt,
        nstout          = nstout,
        nsteps          = nsteps,
        nloops          = nloops,
        t_equil         = t_equil,
        nsteps_equil    = nsteps_equil,
        nloops_equil    = nloops_equil,
        thermostat      = thermostat,
        gamma           = gamma,
        p_shoot         = p_shoot,
        p_shift         = p_shift,
        one_way_shoot   = one_way_shoot,
        one_way_shift   = one_way_shift,
        n_relax         = n_relax,
        n_acqui         = n_acqui,
        n_branch        = n_branch,
        n_dump          = n_dump,
        dump_relax      = dump_relax,
        dump_acqui      = dump_acqui,
        n_replica       = n_replica,
    )


def _expand_parameter(value: Union[list, float]) -> np.ndarray:
    """Expand a scalar or ``[start, end, n_points, ...]`` spec to a NumPy array.

    A plain scalar is wrapped in a zero-dimensional array. A list must have
    length divisible by 3, interpreted as one or more
    ``[start, end, n_points]`` segments; the segments are concatenated,
    de-duplicated, and sorted.

    Args:
        value: Either a scalar float/int or a list of the form
            ``[start, end, n_points, ...]``.

    Returns:
        NumPy array of unique, sorted float values.

    Raises:
        _ValidationError: If a list is provided whose length is not a
            multiple of 3.
    """
    if isinstance(value, list):
        if len(value) % 3 != 0:
            raise _ValidationError(ValueError(),
                "If the data is given as a list, its length should be a multiple of 3.",
                f"current list is {value}.")

        segments = []
        for i in range(0, len(value), 3):
            start, end, n_points = value[i:i+3]
            segments.append(np.linspace(start, end, int(n_points)))

        return np.unique(np.concatenate(segments))
    else:
        return np.array(value)
