"""Scheduler: the sole orchestration hub.

Combines DAG (topology) + StateDB (state) + slurm (execution) to make
scheduling decisions. Branching logic lives here, not in a separate module.
"""

import os
import time

from kapybara.config.schema import SimulationConfig
from kapybara.config.paths import PathManager
from kapybara.state.db import StateDB
from kapybara.orchestrate.dag import DependencyDAG, DAGNode
from kapybara.orchestrate.slurm import (
    construct_sbatch_command,
    submit_job,
    query_job_states,
    set_mpi_environment,
)


class Scheduler:
    """Main TPS scheduling loop.

    Combines :class:`~kapybara.orchestrate.dag.DependencyDAG` (topology),
    :class:`~kapybara.state.db.StateDB` (state), and SLURM helpers
    (execution) to make scheduling decisions. Branching logic and readiness
    evaluation live here, not in a separate module.

    Responsibilities:

    - SLURM state sync via ``squeue --json`` (batch query).
    - Readiness evaluation: DAG traversal + StateDB branching check.
    - Job submission: config → sbatch parameters → SLURM helpers.
    - Failure retry: re-submit only the pending replicas of failed jobs.

    Attributes:
        config: Frozen simulation configuration.
        paths: Path manager for all simulation directories.
        state_db: Central SQLite state tracker.
        dag: Field-dependency topology graph.
        config_path: Absolute path to the YAML config file (passed to workers).
        quiet: If ``True``, suppress non-essential print output.
    """

    def __init__(self, config: SimulationConfig, paths: PathManager,
                 state_db: StateDB, dag: DependencyDAG,
                 config_path: str,
                 quiet: bool = False):
        """Initialise the Scheduler.

        Args:
            config: Frozen :class:`~kapybara.config.schema.SimulationConfig`.
            paths: :class:`~kapybara.config.paths.PathManager` instance.
            state_db: :class:`~kapybara.state.db.StateDB` instance.
            dag: :class:`~kapybara.orchestrate.dag.DependencyDAG` instance.
            config_path: Absolute path to the YAML config file.
            quiet: Suppress resubmission messages when ``True``.
        """
        self.config = config
        self.paths = paths
        self.state_db = state_db
        self.dag = dag
        self.config_path = config_path
        self.quiet = quiet

    # ── Public entry point ──

    def initialize(self) -> None:
        """Register all jobs in StateDB. Must be called once before :meth:`run`.

        Registers prerun jobs (one per temperature) and TPS jobs (one per
        T × field_value combination) with their DAG dependency info. Uses
        ``INSERT OR IGNORE`` so that re-running after partial completion is
        safe.
        """
        dep_map = self.dag.get_dependency_map()

        for T in self.config.T:
            self.state_db.register_prerun_job(T)
            for node in self.dag.all_nodes():
                if node.T == T:
                    parent_fv = node.parent.field_value if node.parent else None
                    self.state_db.register_tps_job(T, node.field_value, parent_fv)

    def run(self) -> None:
        """Enter the main scheduling loop until all TPS jobs complete.

        Each iteration:

        1. Sync SLURM state (mark disappeared jobs as failed).
        2. If all jobs are completed, exit.
        3. If any jobs failed, resubmit them (only their pending replicas).
        4. Otherwise, find and submit newly ready nodes.
        5. Sleep 10 seconds before the next iteration.
        """
        set_mpi_environment()

        while True:
            self._sync_slurm_states()

            if self._is_all_complete():
                break

            failed = self._get_failed_jobs()
            for T, field in failed:
                self._submit_tps_job(T, field)

            if not failed:
                ready = self._find_ready_nodes()
                for node in ready:
                    self._submit_tps_job(node.T, node.field_value)

            time.sleep(10)

    # ── SLURM state sync ──

    def _sync_slurm_states(self) -> None:
        """Batch-query SLURM for running jobs and mark disappeared ones as failed.

        Queries all jobs currently in ``'submitted'`` or ``'running'`` state in
        StateDB. Any job whose ID is absent from ``squeue --json`` output is
        marked as failed (both at the job and replica level).
        """
        prerun_jobs = self.state_db.get_running_prerun_jobs()
        tps_jobs = self.state_db.get_running_tps_jobs()

        all_ids = list(prerun_jobs.values()) + list(tps_jobs.values())
        active = query_job_states(all_ids)

        for T, job_id in prerun_jobs.items():
            if job_id not in active:
                self.state_db.mark_missing_prerun_failed(T)

        for (T, field), job_id in tps_jobs.items():
            if job_id not in active:
                self.state_db.mark_missing_tps_failed(T, field)

    # ── Readiness evaluation (moved from DAG.traverse_ready) ──

    def _find_ready_nodes(self) -> list[DAGNode]:
        """Traverse DAG topology and StateDB to find submittable nodes.

        A node is ready to submit if all of the following hold:

        1. Its job status in StateDB is ``'pending'``.
        2. One of:
           a) It is a root node (no parent) **and** the prerun for its
              temperature has completed (all ``n_replica`` replicas done).
           b) Its parent job has ``'completed'`` status.
           c) Its parent job is ``'submitted'`` or ``'running'`` **and**
              the branching condition is met (all parent replicas have
              ``run_index >= n_branch`` in the ``acqui`` phase).

        Returns:
            List of :class:`~kapybara.orchestrate.dag.DAGNode` objects that
            are ready to be submitted.
        """
        ready = []
        for node in self.dag.all_nodes():
            status = self.state_db.get_tps_job_status(node.T, node.field_value)
            if status != "pending":
                continue

            if node.parent is None:
                if self.state_db.is_prerun_completed(
                    node.T, self.config.n_replica
                ):
                    ready.append(node)
            else:
                parent_status = self.state_db.get_tps_job_status(
                    node.parent.T, node.parent.field_value
                )
                if parent_status == "completed":
                    ready.append(node)
                elif parent_status in ("submitted", "running"):
                    if self.state_db.can_branch_from(
                        node.parent.T, node.parent.field_value,
                        self.config.n_branch, self.config.n_replica,
                    ):
                        ready.append(node)
        return ready

    # ── Job submission ──

    def _submit_tps_job(self, T: str, field_value: str) -> None:
        """Build and submit a SLURM TPS job for one (T, field_value) point.

        Queries which replicas are not yet completed so that resubmissions
        allocate only the CPUs still needed. Passes ``-r <replica_list>``
        to the worker command so that each CPU runs exactly one unfinished
        replica.

        Args:
            T: Temperature string.
            field_value: Field value string.
        """
        pending = self.state_db.get_pending_tps_replicas(
            T, field_value, self.config.n_replica
        )
        if not pending:
            return  # all replicas already completed, nothing to do

        n_pending = len(pending)
        replicas_str = ",".join(str(r) for r in pending)

        worker_cmd = (
            f"srun --mpi=pmi2 python -m kapybara.commands.run"
            f" -c {self.config_path}"
            f" -t {T} -f {field_value} -r {replicas_str}"
        )

        cmd = construct_sbatch_command(
            job_name=f"{self.config.job_name}_{T}_{field_value}",
            partition=self.config.partition,
            n_tasks=n_pending,
            stdout_path=os.path.join(self.paths.step2_stdout, T, f"{field_value}.out"),
            stderr_path=os.path.join(self.paths.step2_stderr, T, f"{field_value}.err"),
            command=worker_cmd,
            exclude=self.config.exclude,
        )
        job_id = submit_job(cmd)
        self.state_db.submit_tps_job(T, field_value, job_id)
        if not self.quiet and n_pending < self.config.n_replica:
            print(f"Scheduler: T={T} f={field_value} — "
                  f"resubmitted {n_pending}/{self.config.n_replica} replicas "
                  f"(job {job_id}).")

    # ── Status checks ──

    def _is_all_complete(self) -> bool:
        statuses = self.state_db.get_all_tps_statuses()
        return all(
            status == "completed"
            for t_statuses in statuses.values()
            for status in t_statuses.values()
        )

    def _get_failed_jobs(self) -> list[tuple[str, str]]:
        statuses = self.state_db.get_all_tps_statuses()
        return [
            (T, field)
            for T, t_statuses in statuses.items()
            for field, status in t_statuses.items()
            if status == "failed"
        ]
