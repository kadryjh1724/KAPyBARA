Configuration Reference
=======================

KAPyBARA is configured via a YAML file passed with ``-c``/``--config`` to all subcommands.
The file is loaded by :func:`kapybara.config.loader.load_config` and validated into a
:class:`kapybara.config.schema.SimulationConfig` frozen dataclass.

Section 1: Basic information
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Field
     - Type
     - Description
   * - ``work_directory``
     - ``str``
     - Absolute path to the working directory where all output is written.
   * - ``job_name``
     - ``str``
     - Base name for SLURM job names and output subdirectories.

Section 2: System size
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Field
     - Type
     - Description
   * - ``n_particles``
     - ``int``
     - Total number of particles. Species counts: ``N_A = 4*n//5``, ``N_B = n//5``.
   * - ``t_obs``
     - ``float``
     - Observation window length (in reduced LJ time units). ``nsteps = t_obs / dt``.
   * - ``density``
     - ``float``
     - Reduced number density. Box size is derived as ``cbrt(n_particles / density)``.
   * - ``n_replica``
     - ``int``
     - Number of parallel replicas per (T, field) point.

Section 3: MD settings
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Field
     - Type
     - Description
   * - ``dt``
     - ``float``
     - MD timestep (reduced LJ units).
   * - ``nstout``
     - ``int``
     - Frames saved per observation window (``nloops = nsteps / nstout``).
   * - ``t_equil``
     - ``float``
     - Equilibration run length before production (LJ time units).
   * - ``thermostat``
     - ``str``
     - Thermostat type: ``"Nose-Hoover"``, ``"Langevin"``, or ``"MSC"``.
   * - ``gamma``
     - ``float``
     - Langevin damping coefficient. Required only when ``thermostat = "Langevin"``.

Section 4: Biasing field
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Field
     - Type
     - Description
   * - ``runtype``
     - ``str``
     - Run mode: ``"g"`` (T-g space), ``"s"`` (T-s space), or ``"sg"``.
   * - ``T``
     - ``[min, max, n]``
     - Temperature grid: linspace from ``min`` to ``max`` with ``n`` points.
   * - ``s``
     - ``float`` or ``[min, max, n]``
     - Counting field *s*. Scalar for a single value.
   * - ``g``
     - ``float`` or ``[min, max, n]``
     - Activity field *g*. Scalar for a single value.
   * - ``n_decimals``
     - ``[int, int, int]``
     - Decimal places for formatting ``[T, s, g]`` as strings (used in file paths).

Section 5: Transition path sampling
-------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Field
     - Type
     - Description
   * - ``p_shoot``
     - ``float``
     - Probability of a shooting move. Must satisfy ``p_shoot + p_shift = 1``.
   * - ``p_shift``
     - ``float``
     - Probability of a shift move.
   * - ``one_way_shoot``
     - ``bool``
     - If ``True``, use one-way (forward-only) shooting.
   * - ``one_way_shift``
     - ``bool``
     - If ``True``, use one-way (forward-only) shifting.
   * - ``n_relax``
     - ``int``
     - Number of TPS moves in the relaxation phase (decorrelation; not saved).
   * - ``n_acqui``
     - ``int``
     - Number of TPS moves in the acquisition phase (data collection).
   * - ``n_branch``
     - ``int``
     - A child field-value node can start once the parent has completed ``n_branch``
       acquisition moves per replica.
   * - ``n_dump``
     - ``[int, int]``
     - Number of checkpoint files saved: ``[n_relax_dumps, n_acqui_dumps]``.
       Dump indices are spaced evenly via linspace.

Section 6: Job submit settings
--------------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Field
     - Type
     - Description
   * - ``partition``
     - ``str``
     - SLURM partition(s) to submit to (comma-separated).
   * - ``exclude``
     - ``list[str]`` or ``null``
     - List of node names to exclude (e.g. ``["node38"]``). Leave empty for none.

Example
-------

.. code-block:: yaml

   # ═══════════════ [1] BASIC INFORMATION ═══════════════
   work_directory  : "/path/to/work"
   job_name        : "example"

   # ══════════════════ [2] SYSTEM SIZE ══════════════════
   n_particles     : 250
   t_obs           : 100
   density         : 1.2
   n_replica       : 16

   # ══════════════════ [3] MD SETTINGS ══════════════════
   dt              : 0.005
   nstout          : 200
   t_equil         : 1000
   thermostat      : "Nose-Hoover"
   gamma           : 10

   # ═════════════════ [4] BIASING FIELD ═════════════════
   runtype         : "g"
   T               : [0.65, 0.66, 6]
   s               : 0.0
   g               : [-0.001, 0.001, 5]
   n_decimals      : [4, 5, 5]

   # ════════════ [5] TRANSITION PATH SAMPLING ═══════════
   one_way_shoot   : False
   one_way_shift   : False
   p_shoot         : 0.15
   p_shift         : 0.85
   n_relax         : 100
   n_acqui         : 500
   n_branch        : 100
   n_dump          : [2, 10]

   # ══════════════ [6] JOB SUBMIT SETTINGS ══════════════
   partition       : "smallmem,largemem"
   exclude         :
