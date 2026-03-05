"""SLURM worker entry-point modules.

Contains the command modules invoked by SLURM via ``sbatch --wrap``:

- ``prerun``: Runs per-temperature prerun replicas (minimize → equilibrate →
  production MD) in parallel via multiprocessing.
- ``run``:    Runs per-(T, field_value) TPS replicas in parallel via
  multiprocessing.
"""
