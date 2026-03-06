Quick Start
===========

This guide walks through the basic KAPyBARA workflow:
**prerun** → **run** → **monitor**.

1. Prepare a config file
------------------------

Create a ``config.yaml`` (see :doc:`configuration` for all fields):

.. code-block:: yaml

   # [1] BASIC INFORMATION
   work_directory  : "/path/to/work"
   job_name        : "example"

   # [2] SYSTEM SIZE
   n_particles     : 250
   t_obs           : 100
   density         : 1.2
   n_replica       : 16

   # [3] MD SETTINGS
   dt              : 0.005
   nstout          : 200
   t_equil         : 1000
   thermostat      : "Nose-Hoover"

   # [4] BIASING FIELD
   runtype         : "g"
   T               : [0.65, 0.66, 6]
   s               : 0.0
   g               : [-0.001, 0.001, 5]
   n_decimals      : [4, 5, 5]

   # [5] TRANSITION PATH SAMPLING
   one_way_shoot   : False
   one_way_shift   : False
   p_shoot         : 0.15
   p_shift         : 0.85
   n_relax         : 100
   n_acqui         : 500
   n_branch        : 100
   n_dump          : [2, 10]

   # [6] JOB SUBMIT SETTINGS
   partition       : "smallmem,largemem"
   exclude         :

2. Pre-run
----------

Generate equilibrated initial trajectories for each temperature *T*:

.. code-block:: bash

   kapybara prerun -c config.yaml

This submits one SLURM job per temperature. Each job spawns ``n_replica``
parallel processes (via ``multiprocessing.spawn``) that each run:
minimize → equilibrate → production.

Progress is tracked in an SQLite database at
``{work_directory}/{job_name}/kapybara.db``.

3. Run TPS
----------

Once all prerun jobs finish, launch transition path sampling:

.. code-block:: bash

   kapybara run -c config.yaml            # foreground (Ctrl+C to stop cleanly)
   kapybara run -c config.yaml --bg       # background — prints PID and log path
   kapybara run -c config.yaml --bg --log /path/to/run.log   # redirect output to custom file

The scheduler polls every 10 seconds, submitting TPS jobs as their
dependencies become ready (branching logic based on ``n_branch``).

With ``--bg``, the scheduler is detached from the terminal and all its output
(job submission messages, completion notices, errors) is captured to a log
file — by default ``{work_directory}/{job_name}/kapybara.log``. Use ``--log``
to redirect to a different path.

To stop a backgrounded scheduler:

.. code-block:: bash

   kapybara stop -c config.yaml           # sends SIGTERM, waits up to 5 s
   kapybara stop -c config.yaml --force   # sends SIGKILL if SIGTERM times out

4. Monitor progress
-------------------

Watch the progress of running jobs in a grid view:

.. code-block:: bash

   kapybara monitor -c config.yaml
   kapybara monitor -c config.yaml -w 30   # refresh every 30 s

Check the SLURM queue for KAPyBARA jobs with progress bars:

.. code-block:: bash

   kapybara queue -c config.yaml
   kapybara queue -c config.yaml -w 10
   kapybara queue -c config.yaml --eta        # show estimated time remaining

Restart behaviour
-----------------

If a job is interrupted, rerunning the same command resumes from the
latest ``.npy`` checkpoint automatically — no manual intervention needed.

5. Analysis (coming soon)
-------------------------

Post-processing and analysis tools (acceptance statistics, path data
extraction, MBAR reweighting) are planned for a future release.
The ``kapybara analysis`` subcommand is reserved for these tools but is
not yet functional in the current version.

.. note::

   Running ``kapybara analysis ...`` will print a ``not yet implemented``
   message and exit without error.
