"""SLURM helpers: sbatch command construction, job submission, squeue --json.

Design principle: this module does NOT know SimulationConfig.
It only knows SLURM. The caller (Scheduler) extracts values from config
and passes them as plain arguments.
"""

import os
import json
import subprocess
from typing import Optional


def construct_sbatch_command(
    job_name: str,
    partition: str,
    n_tasks: int,
    stdout_path: str,
    stderr_path: str,
    command: str,
    exclude: list[str] | None = None,
) -> list[str]:
    """Build an sbatch command argument list.

    Pure function with no side effects. Allocates one SLURM node with
    ``n_tasks`` CPUs (one per replica), exports the full environment,
    and wraps ``command`` via ``--wrap``.

    Args:
        job_name: SLURM job name (``--job-name``).
        partition: SLURM partition (``--partition``).
        n_tasks: Number of CPUs to request (``--cpus-per-task``).
        stdout_path: Path for SLURM stdout log (``--output``).
        stderr_path: Path for SLURM stderr log (``--error``).
        command: Shell command string to run via ``--wrap``.
        exclude: Optional list of node names to exclude (``--exclude``).

    Returns:
        A list of strings suitable for passing to :func:`subprocess.run`.
    """
    cmd = [
        "sbatch",
        f"--job-name={job_name}",
        f"--partition={partition}",
        f"--ntasks=1",
        f"--cpus-per-task={n_tasks}",
        "--nodes=1",
        "--export=ALL",
        f"--output={stdout_path}",
        f"--error={stderr_path}",
    ]
    if exclude:
        cmd.append(f"--exclude={','.join(exclude)}")
    cmd.extend(["--wrap", command])
    return cmd


def submit_job(sbatch_command: list[str]) -> str:
    """Submit a job via sbatch and return the assigned SLURM job ID.

    Args:
        sbatch_command: Argument list as returned by
            :func:`construct_sbatch_command`.

    Returns:
        The SLURM job ID string (e.g. ``"12345"``).

    Raises:
        subprocess.CalledProcessError: If sbatch returns a non-zero exit code.
    """
    result = subprocess.run(
        sbatch_command, capture_output=True, text=True, check=True
    )
    return result.stdout.strip().split()[-1]


def query_job_states(job_ids: list[str]) -> dict[str, str]:
    """Query multiple job states in a single squeue --json call.

    Returns:
        {job_id: state_string}. Jobs not in squeue are absent from the result
        (i.e., they have finished or failed).
    """
    if not job_ids:
        return {}
    result = subprocess.run(
        ["squeue", "--json", "-j", ",".join(job_ids)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return {}
    data = json.loads(result.stdout)
    return {
        str(job["job_id"]): job["job_state"]
        for job in data.get("jobs", [])
    }


def is_job_running(job_id: str) -> bool:
    """Check whether a single job is still present in the SLURM queue.

    Args:
        job_id: SLURM job ID string to query.

    Returns:
        ``True`` if the job appears in ``squeue`` output, ``False`` if it
        has completed, failed, or is otherwise absent.
    """
    states = query_job_states([job_id])
    return job_id in states


def set_mpi_environment() -> None:
    """Set OMP, OMPI, and PMIX environment variables for LAMMPS MPI execution.

    Sets ``OMP_NUM_THREADS=1`` to prevent OpenMP threading conflicts, and
    configures ``OMPI_MCA_btl_vader_single_copy_mechanism``,
    ``SLURM_MPI_TYPE``, ``PMIX_MCA_gds``, and ``PMIX_MCA_pmi_verbose`` for
    stable PMI2-based communication under SLURM.
    """
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OMPI_MCA_btl_vader_single_copy_mechanism"] = "none"
    os.environ["SLURM_MPI_TYPE"] = "pmi2"
    os.environ["PMIX_MCA_gds"] = "hash"
    os.environ["PMIX_MCA_pmi_verbose"] = "1"
