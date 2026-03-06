# KAPyBARA Codebase Reference

> **Target audience:** LLM agents and developers working on this codebase.
> **Purpose:** Fully describes every module, class, and function — accurate enough to rely on without reading source code.
> **Note on installation:** The `kapybara` CLI is installed in a uv-based virtual environment and can be run directly as `kapybara ...`.

---

## 1. Architecture Overview

KAPyBARA runs Transition Path Sampling (TPS) simulations for a **Kob-Andersen binary Lennard-Jones mixture** on HPC clusters via SLURM. It drives LAMMPS through its Python API.

> **Design principle:** Each module owns exactly one domain and never crosses its boundary.

| Module | Knows about |
|--------|-------------|
| `config/` | YAML parsing, field types, path strings |
| `core/` | LAMMPS commands, physics primitives |
| `state/` | SQLite only (no LAMMPS, no SLURM, no DAG) |
| `orchestrate/` | SLURM topology (`dag`), SLURM API (`slurm`), scheduling decisions (`scheduler`) |
| `sampling/` | TPS move logic and per-replica run loops |
| `prepare/` | Prerun LAMMPS workflow |
| `commands/` | Multiprocessing entry points called by SLURM `srun` |
| `cli/` | argparse wiring and thin orchestration calls |
| `utils/` | Shared utilities (conversions, decorators, console, CSV trim) |

---

## 2. Directory Structure

```
kapybara/
├── __init__.py                  # __version__ = "1.0.0"
│
├── config/                      # Configuration management
│   ├── __init__.py              # Exports: SimulationConfig, load_config, PathManager
│   ├── schema.py                # SimulationConfig frozen dataclass
│   ├── loader.py                # YAML → SimulationConfig (validation + derived fields)
│   └── paths.py                 # PathManager: all file/directory paths
│
├── core/                        # Shared simulation primitives
│   ├── __init__.py              # Exports: Thermostat classes, create_thermostat,
│   │                            #          create_lammps_instance, setup_kob_andersen,
│   │                            #          compute_activity, initialize_log_dict
│   ├── thermostat.py            # Thermostat ABC + Nose-Hoover / Langevin / MSC
│   ├── lammps_setup.py          # LAMMPS instance creation + Kob-Andersen system init
│   ├── activity.py              # compute_activity(): mobile particle count
│   └── log_arrays.py            # initialize_log_dict(): trajectory buffer allocation
│
├── state/                       # SQLite state tracking
│   ├── __init__.py              # Exports: StateDB, DBWriter
│   ├── db.py                    # StateDB: 4-table SQLite schema + full API
│   └── writer.py                # DBWriter: single-writer subprocess for concurrent safety
│
├── orchestrate/                 # SLURM orchestration
│   ├── __init__.py              # Exports: DependencyDAG, DAGNode, slurm helpers, Scheduler
│   ├── dag.py                   # DependencyDAG: pure field-dependency topology
│   ├── slurm.py                 # Config-free SLURM helpers (squeue --json)
│   └── scheduler.py             # Scheduler: sole orchestration hub
│
├── sampling/                    # TPS core logic
│   ├── __init__.py
│   ├── moves.py                 # TPSMoves: shooting/shifting with Metropolis-Hastings
│   └── runners/
│       ├── __init__.py
│       ├── runner_base.py       # _RunnerBase: shared relax+acquisition loop (real impl)
│       ├── runner_g.py          # RunnerTg: T-g space (field_axis="g", inherits _RunnerBase)
│       ├── runner_s.py          # RunnerTs: T-s space (field_axis="s", inherits _RunnerBase)
│       └── runner_sg.py         # RunnerSg: stub (NotImplementedError)
│
├── prepare/                     # Prerun workflow
│   ├── __init__.py
│   └── prepare.py               # Prepare: minimize → equilibrate → production MD
│
├── commands/                    # SLURM srun entry points (multiprocessing workers)
│   ├── __init__.py
│   ├── prerun.py                # python -m kapybara.commands.prerun
│   └── run.py                   # python -m kapybara.commands.run
│
├── cli/                         # CLI layer (argparse + thin calls)
│   ├── __init__.py
│   ├── cli.py                   # kapybara main entry point + subcommand dispatch
│   ├── prerun.py                # 'kapybara prerun' handler
│   ├── run.py                   # 'kapybara run' handler (foreground + background)
│   ├── stop.py                  # 'kapybara stop' handler
│   ├── monitor.py               # 'kapybara monitor' handler
│   ├── queue.py                 # 'kapybara queue' handler
│   └── process.py               # find_scheduler_pid() utility
│
└── utils/                       # Utilities
    ├── __init__.py
    ├── convert.py               # Array conversion + string formatting
    ├── decorate.py              # @measureTime decorator
    ├── trim.py                  # CSV restart trimming
    ├── cstring.py               # Rich console output helpers
    └── errors.py                # _ValidationError exception
```

---

## 3. Module Reference

### 3.1 `config/schema.py` — `SimulationConfig`

Frozen dataclass (`@dataclass(frozen=True)`). Instantiated once by `load_config()` and passed everywhere as read-only. All fields below:

| Field | Type | Description |
|-------|------|-------------|
| `job_name` | `str` | SLURM job name prefix |
| `work_directory` | `str` | Base output directory |
| `partition` | `str` | SLURM partition |
| `exclude` | `List[str]` | Nodes to exclude from SLURM (default `[]`) |
| `n_particles` | `int` | Total particle count |
| `N_A` | `int` | Type-A count: `4 * n_particles // 5` |
| `N_B` | `int` | Type-B count: `n_particles - N_A` |
| `density` | `float` | Reduced number density |
| `box_size` | `float` | Cubic box side: `cbrt(n_particles / density)`, rounded to 4 decimals |
| `n_replica` | `int` | Parallel replica count per (T, field) point |
| `runtype` | `str` | `"g"`, `"s"`, or `"sg"` |
| `T`, `s`, `g` | `List[str]` | Formatted parameter value strings |
| `n_decimals` | `List[int]` | 3-element: decimal places for `[T, s, g]` |
| `dt` | `float` | MD timestep |
| `nstout` | `int` | MD steps between output frames |
| `t_obs` | `float` | Observation window length |
| `nsteps` | `int` | `int(t_obs / dt)` |
| `nloops` | `int` | `int(nsteps / nstout)` — frames per trajectory |
| `t_equil` | `float` | Equilibration time |
| `nsteps_equil` | `int` | `int(t_equil / dt)` |
| `nloops_equil` | `int` | `int(nsteps_equil / nstout)` |
| `thermostat` | `str` | `"Nose-Hoover"`, `"Langevin"`, or `"MSC"` |
| `gamma` | `Optional[float]` | Langevin damping coefficient (Langevin only) |
| `p_shoot` | `float` | Shooting move probability |
| `p_shift` | `float` | Shifting move probability (must sum to 1 with `p_shoot`) |
| `one_way_shoot` | `bool` | Forward-only shooting if True |
| `one_way_shift` | `bool` | Forward-only shifting if True |
| `n_relax` | `int` | Relaxation TPS move count |
| `n_acqui` | `int` | Acquisition TPS move count |
| `n_branch` | `int` | Acquisition run threshold for child job branching |
| `n_dump` | `List[int]` | `[relax_dump_count, acqui_dump_count]` |
| `dump_relax` | `List[int]` | 0-indexed relax dump checkpoints from linspace |
| `dump_acqui` | `List[int]` | 0-indexed acqui dump checkpoints from linspace |

---

### 3.2 `config/loader.py`

#### `load_config(config_path: str, quiet: bool = False) → SimulationConfig`
Loads and validates a YAML config file.
1. Converts path to absolute via `os.path.realpath()`
2. Creates two Rich Console instances (stdout, stderr)
3. `yaml.safe_load()` → calls `_parse()` → returns frozen `SimulationConfig`
4. Raises `FileNotFoundError` or `_ValidationError` on failure

#### `_parse(raw, console, console_stderr, quiet) → SimulationConfig`
Validates every field and computes derived ones:
- Warns if `n_particles % 5 != 0` (non-ideal A:B ratio)
- `box_size = round(cbrt(n_particles / density), 4)`
- `nsteps, nloops, nsteps_equil, nloops_equil` via integer division
- Validates `thermostat` is one of three known strings; validates `gamma > 0` if Langevin
- Validates `runtype` ∈ `{"s", "g", "sg"}`; validates field scalar constraints by runtype
- `dump_relax = linspace(0, n_relax, n_dump[0]+1)[1:] - 1` (0-indexed)
- `dump_acqui = linspace(0, n_acqui, n_dump[1]+1)[1:] - 1` (0-indexed)
- Warns if `n_branch - 1` not in `dump_acqui`

#### `_expand_parameter(value) → np.ndarray`
Converts YAML field parameter to a numpy array of unique sorted floats.
- Scalar → single-element array
- List of length `3k`: groups into `(start, end, n)` tuples → `linspace` each → concatenate + deduplicate

---

### 3.3 `config/paths.py` — `PathManager`

All path strings in one place. Never imported by `state/` or `orchestrate/slurm.py`.

#### `__init__(config)`
Builds path attributes from `{work_directory}/{job_name}/`:
```
self.base           = {work_directory}/{job_name}/
self.step1          = base/step1/
self.step1_trj      = step1/trj/          # per-T trajectory outputs
self.step1_ene      = step1/ene/
self.step1_stdout   = step1/stdout/
self.step1_stderr   = step1/stderr/
self.step2          = base/step2/
self.step2_trj      = step2/trj/
self.step2_ene      = step2/ene/
self.step2_csv      = step2/csv/
self.step2_stdout   = step2/stdout/
self.step2_stderr   = step2/stderr/
self.step3          = base/step3/
self.step3_acc      = step3/acc/
self.step3_dat      = step3/dat/
self.step3_mbar     = step3/mbar/
self.step3_stdout   = step3/stdout/
self.step3_stderr   = step3/stderr/
self.db             = base/kapybara.db
```

#### `ensure_directories() → None`
Creates all required directories (idempotent via `exist_ok=True`):
- `step1/trj/{T}/`, `step1/ene/{T}/` for each T; plus stdout/stderr dirs
- `step2/trj/{T}/{field}/`, `step2/ene/{T}/{field}/`, `step2/csv/{T}/{field}/` for each (T, field) pair
  - `runtype="g"`: field is each g value
  - `runtype="s"`: field is each s value
  - `runtype="sg"`: nested `s/g` subdirectories
- `step2/stdout/{T}/`, `step2/stderr/{T}/` for each T
- `step3/acc/`, `step3/dat/`, `step3/mbar/`, step3 stdout/stderr dirs

---

### 3.4 `core/thermostat.py`

Strategy pattern for LAMMPS thermostat management.

#### `Thermostat` (ABC)
- `fix(lmp, T: str) → None` — abstract: apply LAMMPS `fix` commands
- `unfix(lmp) → None` — abstract: remove LAMMPS `fix` commands

#### `NoseHooverThermostat`
- `fix`: `fix 1 all nvt temp T T 1.5` + `fix 2 all momentum 1 linear 1 1 1 rescale`
- `unfix`: removes fix 2, then fix 1

#### `LangevinThermostat(gamma: float)`
Stores `self.gamma`. Each call to `fix` draws a new random seed from numpy.
- `fix`: `fix 1 all langevin T T {gamma} {rand_seed} zero yes` + `fix 2 all nve` + `fix 3 all momentum 1 linear 1 1 1 rescale`
- `unfix`: removes fix 3, 2, 1

#### `MSCThermostat(nstout: int)`
Stores `self.nstout` (rescaling frequency).
- `fix`: `fix 1 all temp/rescale {nstout} T T 0.01 1.0` + `fix 2 all nve` + `fix 3 all momentum 1 linear 1 1 1 rescale`
- `unfix`: removes fix 3, 2, 1

#### `create_thermostat(config) → Thermostat`
Factory: reads `config.thermostat` → returns appropriate concrete instance.
Raises `ValueError` for unknown thermostat type.

---

### 3.5 `core/lammps_setup.py`

#### `create_lammps_instance() → lammps`
Returns `lammps(name="mpi", cmdargs=["-screen", "none", "-log", "none"])`.
Output fully suppressed.

#### `setup_kob_andersen(lmp, config) → None`
Sends LAMMPS commands to build a Kob-Andersen binary LJ system:
1. Three random seeds drawn: `np.random.randint(1e6, size=3)`
2. `units lj`, `atom_style atomic`, `atom_modify map yes`, `pair_style lj/cut 2.5`, `boundary p p p`
3. Creates cubic box of side `config.box_size`; places `N_A` type-1 and `N_B` type-2 atoms at random positions
4. Sets masses to 1.0 for both types
5. Kob-Andersen pair coefficients: AA=(1.0, 1.0), BB=(0.5, 0.88), AB=(1.5, 0.8)
6. Defines groups `typeA` (type 1) and `typeB` (type 2)

---

### 3.6 `core/activity.py`

#### `compute_activity(pos_array: np.ndarray, box_size: float) → int`
Computes TPS activity K: count of (frame, particle) pairs where squared displacement between consecutive frames exceeds 0.09 (= 0.3²).
1. `diff = pos_array[1:] - pos_array[:-1]` — shape `(n_frames-1, n_particles, 3)`
2. Minimum-image PBC: `diff = mod(diff + box_size/2, box_size) - box_size/2`
3. `dist_sq = sum(diff*diff, axis=2)`
4. Returns `int(sum(dist_sq > 0.09))`

---

### 3.7 `core/log_arrays.py`

#### `initialize_log_dict(nloops: int, n_particles: int) → dict`
Returns pre-allocated zeroed arrays:
```python
{
  "pos": zeros((nloops+1, n_particles, 3)),   # positions at each frame
  "vel": zeros((nloops+1, n_particles, 3)),   # velocities at each frame
  "pe":  zeros(nloops+1),                      # potential energy per frame
  "ke":  zeros(nloops+1),                      # kinetic energy per frame
}
```
Frame 0 stores the initial condition; frames 1..nloops are trajectory frames.

---

### 3.8 `state/writer.py` — `DBWriter`

Provides a dedicated single-writer subprocess to safely handle concurrent SQLite writes from multiprocessing workers (prevents WAL corruption).

#### `_writer_loop(db_path: str, queue: MPQueue) → None`
Worker subprocess event loop:
1. Instantiates `StateDB(db_path)` in direct mode
2. Loops: blocks on `queue.get()`
3. On `(method_name, args)` tuple: calls `getattr(db, method_name)(*args)`; exceptions printed to stderr but loop continues
4. Terminates on `None` sentinel

#### `class DBWriter`
**Attributes:** `_db_path`, `queue` (MPQueue), `_process` (mp.Process | None)

| Method | Action |
|--------|--------|
| `__init__(db_path)` | Creates queue, initializes `_process=None` |
| `start()` | Spawns daemon process named `"kapybara-db-writer"` running `_writer_loop` |
| `stop()` | Enqueues `None` sentinel; joins with 60 s timeout; terminates if still alive |

---

### 3.9 `state/db.py` — `StateDB`

Central SQLite state tracker at `{base}/kapybara.db`. Replaces all legacy `.chk` and `.jobID` files.

#### Database settings
- `PRAGMA journal_mode=DELETE` + `PRAGMA synchronous=FULL` (set at init)
- Per-connection: `busy_timeout=30000 ms`
- `row_factory=sqlite3.Row` (dict-like row access)

#### 4-table schema

| Table | Tracks |
|-------|--------|
| `prerun_jobs` | One SLURM job per temperature T |
| `prerun_state` | One row per (T, replica) |
| `tps_jobs` | One SLURM job per (T, field_value); has `progress_at_submit INTEGER DEFAULT 0` |
| `tps_state` | One row per (T, field_value, replica); has `phase TEXT` and `run_index INTEGER` |

**Indexes:** `prerun_jobs(status)`, `prerun_state(T, status)`, `tps_jobs(status)`, `tps_jobs(T, dependency)`, `tps_state(T, field_value, status)`, `tps_state(T, field_value, phase, run_index)`

#### Dual-mode operation
- **Direct mode** (`write_queue=None`): writes go directly to SQLite
- **Queue mode** (write_queue provided): write calls are enqueued to `DBWriter`; reads always direct

#### Construction
`StateDB(db_path, write_queue=None)` — in direct mode, calls `_init_db()` to create schema.

#### `_connect()` context manager
Connects, sets row_factory and busy_timeout, yields connection, commits on success, rolls back on exception.

#### Prerun Job Layer

| Method | Action |
|--------|--------|
| `register_prerun_job(T)` | `INSERT OR IGNORE` with status `"pending"` |
| `submit_prerun_job(T, job_id)` | Set `slurm_job_id`, `status="submitted"`, `submitted_at` |
| `update_prerun_job_status(T, status)` | Update status + `started_at`/`completed_at` timestamps |
| `get_prerun_job(T) → dict\|None` | Fetch full row |

#### Prerun Replica Layer

| Method | Action |
|--------|--------|
| `update_prerun_state(T, replica, status, error=None)` | Upsert via `ON CONFLICT`; COALESCE preserves first timestamps |
| `mark_prerun_completed(T, replica)` | Calls `update_prerun_state(..., "completed")` |
| `is_prerun_completed(T, n_replica) → bool` | True if `count(status="completed") >= n_replica` |
| `all_preruns_completed(temperatures, n_replica) → bool` | True if all T done |
| `get_prerun_statuses() → dict[T, status]` | From `prerun_jobs` table |
| `is_prerun_replica_completed(T, replica) → bool` | Single-row status check |
| `get_pending_prerun_replicas(T, n_replica) → list[int]` | Replica indices in `range(n_replica)` NOT yet completed |

#### TPS Job Layer

| Method | Action |
|--------|--------|
| `register_tps_job(T, field_value, dependency)` | `INSERT OR IGNORE`; dependency = parent field_value or None |
| `submit_tps_job(T, field_value, job_id, progress_at_submit=0)` | Set submitted status + job ID + progress snapshot |
| `update_tps_job_status(T, field_value, status)` | Update status + timestamps |
| `get_tps_job(T, field_value) → dict\|None` | Fetch full row |
| `get_dependency(T, field_value) → str\|None` | Parent field_value or None |

#### TPS Replica Layer

| Method | Action |
|--------|--------|
| `update_tps_state(T, fv, replica, phase, run_index, status, error=None)` | Upsert via `ON CONFLICT` |
| `mark_tps_completed(T, fv, replica)` | Marks replica; if ALL replicas completed → promotes `tps_jobs.status` to `"completed"` |
| `is_tps_replica_completed(T, fv, replica) → bool` | Single-row check |
| `get_pending_tps_replicas(T, fv, n_replica) → list[int]` | Replica indices in `range(n_replica)` NOT yet completed |

#### TPS Read (Scheduler + Monitor)

| Method | Action |
|--------|--------|
| `get_tps_job_status(T, fv) → str` | Single job status; returns `"pending"` if not found |
| `get_all_tps_statuses() → dict[T, dict[fv, status]]` | Full status map |
| `get_tps_replica_progress(T, fv) → list[dict]` | `{replica, phase, run_index, status}` rows ordered by replica |
| `can_branch_from(T, fv, n_branch, n_replica) → bool` | True if `count(phase="acqui" AND run_index >= n_branch) >= n_replica` |

#### SLURM Sync (Scheduler)

| Method | Action |
|--------|--------|
| `get_running_prerun_jobs() → dict[T, job_id]` | Jobs with `status IN ("submitted", "running")` |
| `get_running_tps_jobs() → dict[(T,fv), job_id]` | Same for TPS |
| `mark_missing_prerun_failed(T)` | Sets incomplete replicas + job to `"failed"` with error `"SLURM job disappeared"` |
| `mark_missing_tps_failed(T, fv)` | Same for TPS |

---

### 3.10 `orchestrate/dag.py`

Pure topology. Never imports `StateDB`, `slurm`, or runtime state.

#### `@dataclass DAGNode`
Fields: `T: str`, `field_value: str`, `parent: DAGNode|None`, `children: list[DAGNode]`

#### `DependencyDAG(config)`
Builds field dependency graph on construction.

- `_build(config)` — dispatches by runtype: `"g"` → `_build_field_chain(config, config.g, n_decimals[2])`, `"s"` → `_build_field_chain(config, config.s, n_decimals[1])`, `"sg"` → NotImplementedError
- `_build_field_chain(config, field_values, n_decimals)`:
  1. Converts strings to floats, sorts by `|value|`
  2. `field_value=0` → root node (parent=None)
  3. Each non-zero value → parent is the nearest preceding value with same sign; fallback to 0 if no same-sign predecessor
  4. Example: `[0, 1, -1, 2, -2]` sorted by abs → `0→None, 1→0, -1→0, 2→1, -2→-1`
  5. Creates `DAGNode` for every `(T, field_value)` pair and links children lists

**Topology queries (read-only):**

| Method | Returns |
|--------|---------|
| `get_node(T, fv) → DAGNode\|None` | Single node |
| `get_roots() → list[DAGNode]` | Nodes with `parent=None` |
| `get_parent(T, fv) → DAGNode\|None` | Parent node |
| `get_children(T, fv) → list[DAGNode]` | Child nodes |
| `all_nodes() → list[DAGNode]` | Flat list of all nodes |
| `get_dependency_map() → dict[fv, parent_fv\|None]` | From first T (topology identical across all T) |

---

### 3.11 `orchestrate/slurm.py`

Does NOT import `SimulationConfig`. Accepts plain arguments only.

#### `construct_sbatch_command(job_name, partition, n_tasks, stdout_path, stderr_path, command, exclude=None) → list[str]`
Builds sbatch argument list:
```
sbatch --job-name={name} --partition={partition} --ntasks=1 --cpus-per-task={n_tasks}
       --nodes=1 --export=ALL --output={stdout} --error={stderr}
       [--exclude=node1,node2,...] --wrap "{command}"
```
Pure function, no side effects.

#### `submit_job(sbatch_command) → str`
Runs `subprocess.run(sbatch_command, check=True)`. Returns last whitespace-separated token of stdout (the job ID string).

#### `query_job_states(job_ids) → dict[job_id, state]`
Runs `squeue --json -j {ids}`. Parses JSON `jobs` array → `{job_id_str: state_str}`.
Jobs absent from squeue are absent from result. Returns `{}` on empty input or squeue error.

#### `is_job_running(job_id) → bool`
`job_id in query_job_states([job_id])`.

#### `set_mpi_environment() → None`
Sets: `OMP_NUM_THREADS=1`, `OMPI_MCA_btl_vader_single_copy_mechanism=none`, `SLURM_MPI_TYPE=pmi2`, `PMIX_MCA_gds=hash`, `PMIX_MCA_pmi_verbose=1`.

---

### 3.12 `orchestrate/scheduler.py` — `Scheduler`

The sole orchestration hub. Combines DAG + StateDB + slurm helpers.

#### `__init__(config, paths, state_db, dag, config_path: str, quiet=False)`
Stores all arguments. `config_path` is the absolute YAML path forwarded to worker commands.

#### `initialize() → None`
Registers all jobs in StateDB (safe to call multiple times via `INSERT OR IGNORE`):
- `state_db.register_prerun_job(T)` for each T
- `state_db.register_tps_job(T, fv, parent_fv)` for each DAG node

#### `run() → None`
Main polling loop. Calls `set_mpi_environment()` then loops every 10 s:
1. `_sync_slurm_states()`
2. If `_is_all_complete()` → break
3. Get `_get_failed_jobs()` → if any, call `_submit_tps_job(T, fv)` for each
4. Else: find `_find_ready_nodes()` → call `_submit_tps_job(T, fv)` for each
5. `time.sleep(10)`

#### `_sync_slurm_states() → None`
Batch-queries squeue for all running prerun + TPS job IDs.
For any job ID no longer in squeue: marks it failed in StateDB.

#### `_find_ready_nodes() → list[DAGNode]`
A node is ready if:
- Its TPS status is `"pending"`, **AND**
- Root node: `state_db.is_prerun_completed(T, n_replica)` is True
- Non-root, parent completed: `parent_status == "completed"`
- Non-root, parent running: `parent_status in ("submitted","running")` AND `state_db.can_branch_from(parent_T, parent_fv, n_branch, n_replica)` is True

#### `_submit_tps_job(T, fv) → None`
1. Gets pending replicas: `state_db.get_pending_tps_replicas(T, fv, n_replica)` — if empty, returns
2. Builds replica string `"0,2,4"` and worker command:
   ```
   srun --mpi=pmi2 python -m kapybara.commands.run -c {config_path} -t {T} -f {fv} -r {replicas_str}
   ```
3. Job name: `{job_name}_{T}_{fv}`; CPUs: `len(pending_replicas)` only
4. `_current_progress(T, fv)` → snapshot for ETA baseline
5. `construct_sbatch_command(...)` → `submit_job(cmd)` → `state_db.submit_tps_job(T, fv, job_id, progress_at_submit=progress)`

#### `_current_progress(T, fv) → int`
Sums total completed steps across all replicas:
- completed replica or replica in acqui phase: `n_relax + run_index`
- replica in relax phase: `run_index`

#### `_is_all_complete() → bool`
All values in `get_all_tps_statuses()` equal `"completed"`.

#### `_get_failed_jobs() → list[(T, fv)]`
All `(T, fv)` pairs with `status == "failed"`.

---

### 3.13 `sampling/moves.py` — `TPSMoves`

Holds mutable TPS trajectory state. Move methods take `lmp` as argument (caller owns the LAMMPS instance).

#### `__init__(config, thermostat)`
Sets `self.shoot` and `self.shift` function pointers based on `config.one_way_shoot`/`one_way_shift`.
Initialises `TPS_pos`, `TPS_vel`, `TPS_PE`, `TPS_KE` to `None`; `TPS_K` to `0`. Caller must set these before first move.

#### `run_one_way_shooting(lmp, pt, T, s, g) → tuple`
Draws new Maxwell-Boltzmann velocities at frame `pt`. Runs `nloops` steps forward only. Returns `_accept_or_reject(...)` result.

#### `run_two_way_shooting(lmp, pt, T, s, g) → tuple`
Draws new velocities at `pt`. Runs backward (reversed velocities) to fill frames `[0, pt)`, then forward to fill frames `(pt, nloops]`. Returns `_accept_or_reject(...)` result.

#### `run_one_way_shifting(lmp, pt, T, s, g) → tuple`
Shifts trajectory forward by `pt` frames. Copies frames `[pt, nloops]` to start of new trajectory, uses `TPS_vel[-1]`, runs `pt` new steps at the end. Returns `_accept_or_reject(...)` result.

#### `run_two_way_shifting(lmp, pt, T, s, g) → tuple`
If `pt < nloops/2`: forward shift (extend end). Else: backward shift (extend start with reversed velocities). Returns `_accept_or_reject(...)` result.

#### `_accept_or_reject(log, s, g) → tuple`
Metropolis-Hastings acceptance:
- `cur_E = sum(TPS_PE + TPS_KE)`, `new_E = sum(log["pe"] + log["ke"])`; `dE = new_E - cur_E`
- `cur_K = TPS_K`, `new_K = compute_activity(log["pos"], config.box_size)`; `dK = new_K - cur_K`
- `r = uniform(0, 1)`, `factor = exp(-float(s)*dK - float(g)*dE)`
- Accept if `r < factor` → update `TPS_pos/vel/PE/KE/K`; reject otherwise
- Returns `(r, cur_E, cur_K, new_E, new_K, dE, dK, factor, accepted)` (9-tuple)

---

### 3.14 `sampling/runners/runner_base.py` — `_RunnerBase`

**The real implementation** of the relax + acquisition TPS loop. `RunnerTg` and `RunnerTs` are thin subclasses.

#### Class attribute: `field_axis`
Must be `"g"` or `"s"` in subclasses. Determines which field is the scan axis.

#### `__init__(config, paths, state_db, moves)`

#### `run(T: str, field_value: str, replica_index: int) → None`
Full execution pipeline:

**1. Field assignment:**
- If `field_axis == "g"`: `g = field_value`, `s = config.s[0]`
- If `field_axis == "s"`: `s = field_value`, `g = config.g[0]`

**2. Restart detection** (scan `.npy` dump files, newest first):
- **Case I**: acqui dump found → skip relax phase, resume acquisition from `restart_idx + 1`
- **Case II**: relax dump found (and no acqui) → resume relax from `restart_idx + 1` (or start acqui if last relax dump)
- **Case III**: no dumps → fresh start

**3. Initial trajectory loading:**
- Fresh start, root (no dependency): `np.load(step1_trj/{T}/{r:02d}.npy)` and matching `.ene`
- Fresh start, non-root: `np.load(step2_trj/{T}/{parent_field}/acqui_{r:02d}_{n_branch-1}.npy)`
- Checkpoint resume: load from found dump file; call `trim_csv()` to truncate CSV

**4. Initialize TPSMoves state:**
```python
moves.TPS_pos = pro_trj[:, :, :3]    # positions
moves.TPS_vel = pro_trj[:, :, 3:]    # velocities
moves.TPS_PE  = pro_ene[0]            # PE row
moves.TPS_KE  = pro_ene[1]            # KE row
moves.TPS_K   = compute_activity(...) # activity
```

**5. Pre-generate random sequences** (before LAMMPS work):
```python
pts_relax = np.random.randint(1, nloops-1, size=n_relax)
pts_acqui = np.random.randint(1, nloops-1, size=n_acqui)
run_relax = np.random.uniform(size=n_relax)
run_acqui = np.random.uniform(size=n_acqui)
```

**6. LAMMPS setup**: `create_lammps_instance()` + `setup_kob_andersen()` + `neigh_modify every 1 delay 5 check yes`

**7. Relaxation loop** (`n_relax` iterations):
- `_move(lmp, pt, p, T, s, g)` → `(stype, result, elapsed)`; `_write_csv(...)`
- At each `dump_relax[i]`: save `.npy` trajectory + energy; `state_db.update_tps_state(..., phase="relax", run_index=idx+1)`

**8. Acquisition loop** (`n_acqui` iterations, same pattern):
- At each `dump_acqui[i]`: save `acqui_*.npy`; `state_db.update_tps_state(..., phase="acqui", run_index=idx+1)`

**9. Completion**: `state_db.mark_tps_completed(T, fv, replica_index)`

**Error handling**: exception → `state_db.update_tps_state(..., "failed")`; `finally: lmp.close()`

**`run_index` convention:** stored as `idx + 1` (1-indexed count of completed runs). `can_branch_from` checks `run_index >= n_branch`. Branching file uses 0-indexed dump name: `acqui_{r:02d}_{n_branch-1}.npy`.

#### `_move(lmp, pt, p, T, s, g) → (stype, result, elapsed)`
Dispatches to `moves.shoot` if `p < config.p_shoot`, else `moves.shift`. Times with `perf_counter`. Returns string type `"SHOOT"` or `"SHIFT"`, 9-tuple result, and elapsed seconds.

#### `_write_csv(csv_path, run_type, idx, pt, stype, result, elapsed) → None` (static)
Appends one line to per-replica CSV. Unpacks result: `(r, cur_E, cur_K, new_E, new_K, dE, dK, factor, accepted)`.

---

### 3.15 `sampling/runners/runner_g.py` — `RunnerTg`

```python
class RunnerTg(_RunnerBase):
    field_axis = "g"
```
Scans g-field (energy bias axis). s is fixed at `config.s[0]`. All logic inherited from `_RunnerBase`.

---

### 3.16 `sampling/runners/runner_s.py` — `RunnerTs`

```python
class RunnerTs(_RunnerBase):
    field_axis = "s"
```
Scans s-field (activity bias axis). g is fixed at `config.g[0]`. All logic inherited from `_RunnerBase`. **Not a stub — fully functional.**

---

### 3.17 `sampling/runners/runner_sg.py` — `RunnerSg`

Stub only. `run(T, field_value, replica_index)` raises `NotImplementedError`.

---

### 3.18 `prepare/prepare.py` — `Prepare`

Prerun workflow: minimize → equilibrate → production MD.

#### `__init__(config, paths, state_db)`

#### `prerun(T: str, replica_index: int) → None`
1. `create_thermostat(config)`, `create_lammps_instance()`, `setup_kob_andersen()`
2. If thermostat is MSC: pre-initialize velocities from Maxwell-Boltzmann
3. `_run_minimize(lmp)` → `t1`; `_run_equilibration(lmp, T, thermostat)` → `t2`; `_run_production(lmp, T, thermostat, replica_index)` → `t3`
4. Prints: `"{replica:02d},{t1:.2f},{t2:.2f},{t3:.2f}"`
5. `state_db.mark_prerun_completed(T, replica_index)`
6. Exception → `state_db.update_prerun_state(..., "failed")`; `finally: lmp.close()`

#### `_run_minimize(lmp) → float`
`minimize 1.0e-4 1.0e-6 1000 10000`. Returns elapsed seconds.

#### `_run_equilibration(lmp, T, thermostat) → float`
`reset_timestep 0` → `thermostat.fix()` → configure thermo output (step, temp, ke, pe, etotal, press every nstout) → `run nsteps_equil` → `thermostat.unfix()`. Returns elapsed seconds.

#### `_run_production(lmp, T, thermostat, replica_index) → float`
Collects `nloops+1` frames via `initialize_log_dict()`. Frame 0: `gather_atoms("x"/"v")` + `get_thermo("pe"/"ke")`. Then loops `nloops` times: `run nstout`, gather frame `i+1`. Saves:
- `step1/trj/{T}/{r:02d}.npy` — shape `(nloops+1, n_particles, 6)`: `concatenate(pos, vel, axis=2)`
- `step1/ene/{T}/{r:02d}.npy` — shape `(2, nloops+1)`: `vstack(pe, ke)`

Returns elapsed seconds.

---

### 3.19 `commands/prerun.py`

**Entry point for SLURM srun.** Arguments: `-c config`, `-t temperature`, `-r replicas` (comma-separated), `[-q]`.

**Architecture:** Single-writer pattern — one `DBWriter` process owns all SQLite writes; worker processes send via queue.

**`if __name__ == "__main__":`**
1. `mp.set_start_method("spawn", force=True)`
2. `load_config()`, `PathManager()`, ensure StateDB schema exists
3. Create direct-mode `StateDB`, update prerun job status to `"running"`, instantiate and start `DBWriter`
4. Spawn `n_replica` processes calling `_run_replica(i, writer.queue)`, join all
5. Stop `DBWriter`; if all replicas done: mark job `"completed"`

**`_run_replica(replica_index, write_queue)`**
1. `load_config()` + `PathManager()` + `StateDB(db_path, write_queue)` (queue mode)
2. `state_db.is_prerun_replica_completed(T, replica)` → skip if True
3. `Prepare(config, paths, state_db).prerun(T, replica_index)`

---

### 3.20 `commands/run.py`

**Entry point for SLURM srun.** Arguments: `-c config`, `-t temperature`, `-f field_value`, `-r replicas`, `[-q]`.

**Architecture:** Same single-writer pattern as `commands/prerun.py`.

**`if __name__ == "__main__":`**
1. `mp.set_start_method("spawn", force=True)`
2. `load_config()`, `PathManager()`, ensure StateDB schema, direct-mode `StateDB`, update TPS job `"running"`, start `DBWriter`
3. Spawn processes calling `_run_replica(i, writer.queue)`, join all, stop `DBWriter`

**`_run_replica(replica_index, write_queue)`**
1. `load_config()` + `PathManager()` + queue-mode `StateDB`
2. Skip if `is_tps_replica_completed(T, field, replica)` is True
3. `create_thermostat(config)` + `TPSMoves(config, thermostat)` + `_create_runner(...)` → `runner.run(T, field, replica_index)`

**`_create_runner(config, paths, state_db, moves)`**
`match config.runtype`: `"g"` → `RunnerTg`, `"s"` → `RunnerTs`, `"sg"` → `RunnerSg`.

---

### 3.21 `cli/cli.py` — `main()`

Entry point (`kapybara = "kapybara.cli:main"` in pyproject.toml). argparse dispatch to subcommands:

| Subcommand | Arguments | Handler |
|------------|-----------|---------|
| `prerun` | `-c`, `-q` | `cli/prerun.py:prerun()` |
| `run` | `-c`, `-q`, `--bg`, `--log` | `cli/run.py:run()` |
| `stop` | `-c`, `--force` | `cli/stop.py:stop()` |
| `monitor` | `-c`, `-w` | `cli/monitor.py:monitor()` |
| `queue` | `-c`, `-w`, `-n` (default 20), `--eta` | `cli/queue.py:queue()` |
| `analysis` | sub-subcommands (not yet implemented) | — |

---

### 3.22 `cli/prerun.py`

**`prerun(args) → None`**
1. `set_mpi_environment()`
2. `load_config()` + `PathManager()` + `StateDB()`; `paths.ensure_directories()`
3. For each T:
   - `get_pending_prerun_replicas(T, n_replica)` → skip T if empty
   - `state_db.register_prerun_job(T)`
   - Worker command: `srun --mpi=pmi2 python -m kapybara.commands.prerun -c {config} -t {T} -r {replicas_str}`
   - Job name: `{job_name}-prerun-{T}`; CPUs: `len(pending_replicas)`
   - `construct_sbatch_command(...)` → `submit_job(cmd)` → `state_db.submit_prerun_job(T, job_id)`
   - Prints: `"kapybara prerun: T={T} — submitted job {job_id} ({n_pending} CPUs)."`

---

### 3.23 `cli/run.py`

**`run(args) → None`**
- If `args.bg`: calls `_run_background(args, paths)` (parent exits after printing PID/log)
- Else: calls `_run_foreground(args, config, paths)`

**`_run_background(args, paths)`**
Spawns detached child: `subprocess.Popen(["python", "-m", "kapybara.cli.cli", "run", "-c", config_path, "-q"], start_new_session=True, stdout=log_file, stderr=log_file)`. Prints PID and log path, then exits.

Default log path: `{base}/kapybara.log`. Override with `--log`.

**`_run_foreground(args, config, paths)`**
1. `paths.ensure_directories()`
2. `StateDB()`, `DependencyDAG()`, `Scheduler(config, paths, state_db, dag, config_path)`
3. Installs `SIGTERM` and `SIGINT` handlers that raise `ShutdownRequested`
4. `scheduler.initialize()` → `scheduler.run()`
5. On `ShutdownRequested`: prints `"kapybara: scheduler stopped by signal."`

**`class ShutdownRequested(Exception)`** — raised by signal handlers to break scheduler loop cleanly.

---

### 3.24 `cli/stop.py`

**`stop(args) → None`**
1. `find_scheduler_pid(args.config)` — if None: prints error, exits with code 1
2. `os.kill(pid, SIGTERM)` → prints `"kapybara: sent SIGTERM to PID {pid}."`
3. Polls 10 × 0.5 s using `os.kill(pid, 0)` (existence check)
   - `ProcessLookupError` → process exited, prints `"kapybara: scheduler stopped."`, returns
4. After 5 s: if `--force` → `os.kill(pid, SIGKILL)`, prints `"kapybara: sent SIGKILL."`; else prints instructions

---

### 3.25 `cli/process.py`

**`find_scheduler_pid(config_path: str) → int | None`**
Locates running `kapybara run` process via `ps aux | grep 'kapybara.*{config_name}'`.
Excludes grep itself and monitor/queue/stop subprocesses. Returns first PID found, or None.

---

### 3.26 `cli/monitor.py`

**`monitor(args) → None`**
1. `load_config()` + `PathManager()` + `StateDB()`
2. Loop: `os.system("clear")` in watch mode → `_print_board(config, state_db, args)` → `time.sleep(args.watch)` or break

**`_print_board(config, state_db, args)`**
Renders ASCII (T × field) progress board. Places larger dimension on x-axis. Each cell shows 2-digit status. Calls `_cell_status(state_db, T, fv, n_replica)` per cell.

**`_cell_status(state_db, T, fv, n_replica)`**
Queries `get_tps_job_status(T, fv)`:
- `"completed"` → blue cell
- `"submitted"` or `"running"` → green cell with count of remaining (incomplete) replicas
- `"failed"` → red cell
- else → black cell (pending/not started)

---

### 3.27 `cli/queue.py`

**`queue(args) → None`**
1. `load_config()` + `PathManager()` + `StateDB()`
2. Loop: `squeue --format "%i %P %j %M %R" | grep {job_name} | grep -v prerun`
3. Box-drawing table with dynamic column widths based on `config.n_decimals`
4. Extracts `T = job_name.split("_")[-2]`, `field_value = job_name.split("_")[-1]`
5. Progress: `_job_progress(state_db, T, fv, n_relax)`
6. Optional ETA column (with `--eta`): `_compute_eta(state_db, T, fv, dt_secs, progress, total)`
7. PENDING jobs show `-` placeholders

**`_job_progress(state_db, T, fv, n_relax) → int`**
Sums approximate completed steps from `get_tps_replica_progress()`:
- completed or acqui-phase replica: `n_relax + run_index`
- relax-phase replica: `run_index`

**`_compute_eta(state_db, T, fv, dt_secs, progress, total) → float | None`**
Uses `progress_at_submit` (from `tps_jobs` table) to isolate the rate of the current SLURM job after kill/resubmit. Returns None if progress < 1%, no elapsed time, or no progress delta.

**`_progress_color(progress, total, colormap) → str`**
Returns RdYlGn gradient colored 3-character percentage string.

---

### 3.28 `utils/convert.py`

| Function | Description |
|----------|-------------|
| `npy2Cint(arr)` | numpy int32 → ctypes `POINTER(c_int)` for LAMMPS scatter |
| `npy2Cdouble(arr)` | numpy float64 → ctypes `POINTER(c_double)` for LAMMPS scatter |
| `str2npy(strList)` | `List[str]` → `np.ndarray` float64 |
| `npy2str(arr, n)` | `np.ndarray` → `List[str]` formatted to `n` decimal places |
| `strfdelta_short(delta_t)` | Float seconds → `"DDd HHh MMm"` (11 chars, no seconds) |
| `strfdelta2(delta_t)` | Float seconds → `"DDd HHh MMm SSs"` (15 chars) |

---

### 3.29 `utils/decorate.py`

**`@measureTime`** — decorator. Wraps function, measures wall time via `perf_counter`, returns `(original_result, elapsed_seconds)`. Preserves metadata via `functools.wraps`.

---

### 3.30 `utils/trim.py`

**`trim_csv(csv_path: str, restart_idx: int, run_type: str) → None`**
Reads CSV with pandas. Finds row where `RUN_TYPE == run_type AND RUN_IDX == str(restart_idx)`. Truncates to that row (inclusive) and overwrites file. Raises `ValueError` if no matching row found.

---

### 3.31 `utils/cstring.py`

All three functions embed caller's qualified name in panel title via `sys._getframe(1)`.

| Function | Panel color | Description |
|----------|-------------|-------------|
| `prettyNotification(console, msg1, msg2=None)` | bright_yellow | Stdout notification panel |
| `prettyWarning(console_stderr, msg1, msg2)` | orange_red1 | Stderr warning panel |
| `prettyError(console_stderr, exc, msg1, msg2=None)` | red3 | Stderr error panel with exception type |

---

### 3.32 `utils/errors.py`

**`class _ValidationError(Exception)`**
Wraps a base exception with human-readable messages for the config validation pipeline.
- `exc`: underlying exception
- `message1`: primary error description
- `message2`: optional secondary detail

---

## 4. CLI Command Call Flows

### 4.1 `kapybara prerun -c config.yaml`

```
cli/cli.py: main()
  └─ cli/prerun.py: prerun(args)
       ├─ set_mpi_environment()
       ├─ load_config() + PathManager() + StateDB()
       ├─ paths.ensure_directories()
       └─ for T in config.T:
            ├─ state_db.get_pending_prerun_replicas(T, n_replica)  → skip if empty
            ├─ state_db.register_prerun_job(T)
            ├─ construct_sbatch_command(...)            [orchestrate/slurm.py]
            ├─ submit_job(cmd) → job_id
            └─ state_db.submit_prerun_job(T, job_id)

── SLURM executes: srun python -m kapybara.commands.prerun -c config -t T -r "0,1,2" ──

commands/prerun.py: __main__
  ├─ mp.set_start_method("spawn")
  ├─ load_config() + PathManager() + StateDB (direct) + update_prerun_job_status("running")
  ├─ DBWriter.start()
  └─ mp.Process(target=_run_replica, args=(i, writer.queue)) × len(replicas)
       └─ _run_replica(replica_index, write_queue)
            ├─ load_config() + PathManager() + StateDB(queue_mode)
            ├─ is_prerun_replica_completed() → skip if True
            └─ Prepare(config, paths, state_db).prerun(T, replica_index)
                 ├─ create_lammps_instance() + setup_kob_andersen()
                 ├─ create_thermostat()
                 ├─ _run_minimize(lmp)
                 ├─ _run_equilibration(lmp, T, thermostat)
                 ├─ _run_production(lmp, T, thermostat, r)
                 │    └─ np.save(step1/trj/{T}/{r:02d}.npy), np.save(step1/ene/...)
                 └─ state_db.mark_prerun_completed(T, r)
```

---

### 4.2 `kapybara run -c config.yaml`

```
cli/cli.py: main()
  └─ cli/run.py: run(args)
       └─ _run_foreground(args, config, paths)
            ├─ load_config() + PathManager() + StateDB()
            ├─ DependencyDAG(config)  → _build_field_chain()
            ├─ Scheduler(config, paths, state_db, dag, config_path)
            ├─ scheduler.initialize()
            │    ├─ state_db.register_prerun_job(T) × n_T
            │    └─ state_db.register_tps_job(T, fv, parent_fv) × all nodes
            └─ scheduler.run()                          [installs SIGTERM/SIGINT handlers]
                 └─ set_mpi_environment()
                 └─ while True:
                      ├─ _sync_slurm_states()
                      │    ├─ state_db.get_running_{prerun,tps}_jobs()
                      │    ├─ query_job_states(all_ids)
                      │    └─ mark_missing_*_failed() for absent IDs
                      ├─ _is_all_complete() → break if True
                      ├─ _get_failed_jobs() → resubmit if any
                      ├─ _find_ready_nodes()
                      │    └─ for node in dag.all_nodes():
                      │         ├─ get_tps_job_status() == "pending" ?
                      │         ├─ [root] is_prerun_completed(T, n_replica) ?
                      │         ├─ [non-root, parent done] parent status == "completed" ?
                      │         └─ [non-root, branching] can_branch_from(parent, n_branch) ?
                      ├─ _submit_tps_job(T, fv)
                      │    ├─ get_pending_tps_replicas(T, fv, n_replica)
                      │    ├─ construct_sbatch_command(...)
                      │    ├─ submit_job(cmd) → job_id
                      │    └─ state_db.submit_tps_job(T, fv, job_id, progress_at_submit)
                      └─ time.sleep(10)

── SLURM executes: srun python -m kapybara.commands.run -c config -t T -f fv -r "0,1,2" ──

commands/run.py: __main__
  ├─ mp.set_start_method("spawn")
  ├─ StateDB (direct) + update_tps_job_status("running") + DBWriter.start()
  └─ mp.Process(target=_run_replica, args=(i, writer.queue)) × len(replicas)
       └─ _run_replica(replica_index, write_queue)
            ├─ load_config() + PathManager() + StateDB(queue_mode)
            ├─ is_tps_replica_completed() → skip if True
            ├─ create_thermostat() + TPSMoves(config, thermostat)
            ├─ _create_runner() → RunnerTg / RunnerTs / RunnerSg
            └─ runner.run(T, field_value, replica_index)     [sampling/runners/runner_base.py]
                 ├─ [Restart detection: scan .npy files]
                 ├─ [Load initial trajectory]
                 │    ├─ (root) np.load(step1/trj/{T}/{r:02d}.npy)
                 │    └─ (non-root) np.load(step2/trj/{T}/{g_dep}/acqui_{r}_{n_branch-1}.npy)
                 ├─ moves.TPS_pos/vel/PE/KE/K ← loaded data
                 ├─ create_lammps_instance() + setup_kob_andersen()
                 ├─ [Relaxation loop: n_relax iterations]
                 │    └─ _move() → moves.shoot/shift()
                 │         ├─ thermostat.fix/unfix + lmp run + gather_atoms
                 │         └─ _accept_or_reject() → compute_activity()
                 │    [at dump_relax[i]]: np.save() + update_tps_state(phase="relax", run_index=idx+1)
                 ├─ [Acquisition loop: n_acqui iterations, same pattern]
                 │    [at dump_acqui[i]]: np.save(acqui_*.npy) + update_tps_state(phase="acqui")
                 └─ state_db.mark_tps_completed(T, fv, r)
```

---

### 4.3 `kapybara run -c config.yaml --bg`

```
cli/run.py: _run_background(args, paths)
  ├─ Opens log file (--log or {base}/kapybara.log)
  ├─ subprocess.Popen(["python","-m","kapybara.cli.cli","run","-c",config_path,"-q"],
  │                   start_new_session=True, stdout=log_file, stderr=log_file)
  ├─ Prints "kapybara: backgrounded (PID {pid})"
  ├─ Prints "kapybara: log → {log_path}"
  └─ Parent exits; child becomes detached daemon

kapybara stop -c config.yaml [--force]
  └─ cli/stop.py: stop(args)
       ├─ find_scheduler_pid(config_path) → pid
       ├─ os.kill(pid, SIGTERM)
       ├─ Poll 10 × 0.5 s via os.kill(pid, 0)
       │    └─ ProcessLookupError → "kapybara: scheduler stopped."
       └─ After 5 s: --force → SIGKILL, else → print instructions
```

---

### 4.4 `kapybara monitor -c config.yaml [-w 30]`

```
cli/monitor.py: monitor(args)
  ├─ load_config() + PathManager() + StateDB()
  └─ while True:
       ├─ os.system("clear")  [watch mode only]
       ├─ _print_board(config, state_db, args)
       │    ├─ find_scheduler_pid(args.config)  [show running PID]
       │    └─ for (T, field_value) in grid:
       │         └─ _cell_status(state_db, T, fv, n_replica)
       │              ├─ get_tps_job_status(T, fv)
       │              └─ [running] get_tps_replica_progress(T, fv)
       └─ time.sleep(args.watch) or break
```

---

### 4.5 `kapybara queue -c config.yaml [--eta]`

```
cli/queue.py: queue(args)
  ├─ load_config() + PathManager() + StateDB()
  └─ while True:
       ├─ squeue --format "%i %P %j %M %R" | grep job_name | grep -v prerun
       ├─ Print RdYlGn colorbar header
       ├─ for squeue_line in lines:
       │    ├─ T = job_name.split("_")[-2], field_value = job_name.split("_")[-1]
       │    ├─ _job_progress(state_db, T, fv, n_relax)
       │    ├─ [--eta] _compute_eta(state_db, T, fv, elapsed_secs, progress, total)
       │    └─ _progress_color(progress, total)
       └─ time.sleep(args.watch) or break
```

---

## 5. Data Flow: Trajectory Files

```
step1/trj/{T}/{r:02d}.npy        shape: (nloops+1, n_particles, 6)   pos[:,:,:3] | vel[:,:,3:]
step1/ene/{T}/{r:02d}.npy        shape: (2, nloops+1)                row 0: PE, row 1: KE

step2/trj/{T}/{field}/relax_{r:02d}_{idx}.npy    (same shape as step1 trj)
step2/trj/{T}/{field}/acqui_{r:02d}_{idx}.npy    idx ∈ dump_acqui (0-indexed)
step2/ene/{T}/{field}/relax_{r:02d}_{idx}.npy    (same shape as step1 ene)
step2/ene/{T}/{field}/acqui_{r:02d}_{idx}.npy
step2/csv/{T}/{field}/{r:02d}.csv                one row per TPS move
```

**CSV columns:**
`RUN_TYPE, SAMPLING_TYPE, RUN_IDX, TPS_PNT, RND_NUM, CUR_E, CUR_K, NEW_E, NEW_K, ΔE, ΔK, exp(-sΔK-gΔE), STATUS, RUN_TIME`

---

## 6. Key Design Invariants

| Invariant | Where enforced |
|-----------|---------------|
| `SimulationConfig` is immutable after creation | `@dataclass(frozen=True)` |
| `slurm.py` never imports `SimulationConfig` | Module boundary |
| `dag.py` never imports `StateDB` or `slurm` | Module boundary |
| `state/db.py` never imports LAMMPS or SLURM | Module boundary |
| All scheduling decisions in `Scheduler` | Single responsibility |
| SQLite uses `journal_mode=DELETE`, `synchronous=FULL`, `busy_timeout=30000 ms` | `state/db.py._init_db()` |
| Single SQLite writer via `DBWriter` subprocess under multiprocessing | `state/writer.py` |
| `run_index` in `tps_state` is 1-indexed count of completed runs | `runner_base.py: idx + 1` |
| Branching file: `acqui_{r}_{n_branch-1}.npy` (0-indexed dump name) | `runner_base.py` |
| `can_branch_from` checks `run_index >= n_branch` (1-indexed) | `state/db.py` |
| SLURM TPS job name format: `{job_name}_{T}_{field_value}` | `scheduler.py` |
| Prerun job name format: `{job_name}-prerun-{T}` | `cli/prerun.py` |
| Background process detected by `find_scheduler_pid()` via `ps aux \| grep` | `cli/process.py` |
| Only pending replicas submitted on retry (minimal CPU allocation) | `scheduler.py`, `commands/` |
| `RunnerTg` and `RunnerTs` share all logic via `_RunnerBase` | `runner_base.py` |
| `RunnerSg` is not implemented (raises `NotImplementedError`) | `runner_sg.py` |

---

## 7. Dependencies (pyproject.toml)

- Python >= 3.12
- `matplotlib >= 3.10.8`
- `numpy >= 2.4.2`
- `pandas >= 3.0.1`
- `pyyaml >= 6.0.3`
- `rich >= 14.3.3`
- External (not pip-installable): LAMMPS Python API (must be built as shared library)
