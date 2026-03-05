KAPyBARA Documentation
======================

**KAPyBARA** is a Kob-Andersen model simulation package with Python-Based TRAjectory sampling, powered by `LAMMPS <https://github.com/lammps/lammps>`_.

Key features:

- One-line initial properly equilibrated trajectory generation
- One-line transition path sampling (TPS)
- Automated massive job submission to SLURM clusters
- Automated job monitoring and restarting
- SQLite-backed state tracking with automatic restart support

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   configuration

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/config
   api/core
   api/state
   api/orchestrate
   api/sampling
   api/prepare
   api/commands
   api/cli
   api/utils

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
