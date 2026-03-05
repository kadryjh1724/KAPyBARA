Installation
============

Requirements
------------

- Python 3.12+
- LAMMPS (shared build with Python API enabled)
- SLURM-based HPC cluster
- MPI (e.g. OpenMPI or Intel MPI)

.. note::
   KAPyBARA drives LAMMPS through its Python API (``from lammps import lammps``),
   so LAMMPS must be compiled in **shared library mode** before installing KAPyBARA.
   This is the trickiest part of the setup and is covered in detail below.

----

Step 1: Set up a virtual environment
-------------------------------------

Choose either conda or uv. Both work, but uv is recommended for cleaner
dependency management.

Option A — conda
^^^^^^^^^^^^^^^^

.. code-block:: bash

   conda create -n kapybara python=3.12
   conda activate kapybara

Option B — uv (recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Install uv if you haven't already (see `uv docs <https://docs.astral.sh/uv/>`_).

.. code-block:: bash

   uv venv
   source .venv/bin/activate

.. note::
   With uv you can skip ``source .venv/bin/activate`` entirely and use
   ``uv run kapybara ...`` / ``uv run python ...`` to run inside the venv.

----

Step 2: Build and install LAMMPS
----------------------------------

Clone the latest LAMMPS release:

.. code-block:: bash

   git clone -b release https://github.com/lammps/lammps.git /path/to/lammps
   cd /path/to/lammps

Build in **shared mode** — this is required for the Python API:

.. code-block:: bash

   mkdir build && cd build
   cmake ../cmake \
       -D BUILD_SHARED_LIBS=yes \
       -D BUILD_MPI=yes \
       -D BUILD_OMP=yes \
       -D LAMMPS_MACHINE=mpi
   make -j$(nproc)
   make install

.. note::
   CMake options may vary depending on your cluster's MPI/compiler setup.
   See the `LAMMPS CMake build guide <https://docs.lammps.org/Build_cmake.html>`_
   for all available options.

Install the Python bindings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The procedure differs between conda and uv.

**conda:** use ``make install-python`` (installs into the active conda env):

.. code-block:: bash

   make install-python

**uv:** do *not* use ``make install-python`` — it invokes pip outside the venv.
Instead, install the wheel that CMake produced in the build directory:

.. code-block:: bash

   # Still inside lammps/build/
   uv pip install lammps-*.whl

The wheel filename looks like ``lammps-2025.7.22-cp312-cp312-linux_x86_64.whl``.
The exact name depends on the LAMMPS version and your platform.

----

Step 3: Verify the LAMMPS installation
----------------------------------------

Run this quick check (make sure the venv is active):

.. code-block:: bash

   # conda
   python -c "from lammps import lammps; lmp = lammps(name='mpi'); lmp.close()"

   # uv (venv not activated)
   uv run python -c "from lammps import lammps; lmp = lammps(name='mpi'); lmp.close()"

Expected output (LAMMPS version will vary):

.. code-block:: text

   LAMMPS (22 Jul 2025 - Update 3)
   OMP_NUM_THREADS environment is not set. Defaulting to 1 thread. (src/comm.cpp:99)
       using 1 OpenMP thread(s) per MPI task
   Total wall time: 0:00:00

.. tip::
   Add ``export OMP_NUM_THREADS=1`` to your SLURM job script to suppress the
   ``OMP_NUM_THREADS`` warning at runtime.

----

Step 4: Install KAPyBARA
-------------------------

Clone and install:

.. code-block:: bash

   git clone https://github.com/kadryjh1724/kapybara.git /path/to/KAPyBARA
   cd /path/to/KAPyBARA

   # conda
   pip install .

   # uv
   uv pip install .

Verify the CLI is available:

.. code-block:: bash

   kapybara --help
   # or with uv (from within the KAPyBARA directory): uv run kapybara --help

----

Step 5: Make ``kapybara`` available from anywhere
---------------------------------------------------

After installing, the ``kapybara`` executable lives inside the virtual
environment's ``bin/`` directory. By default ``uv run kapybara`` only works
from inside the KAPyBARA project directory, because ``uv run`` looks for the
nearest ``pyproject.toml`` to locate the venv.

In practice you will run KAPyBARA from wherever your ``.yaml`` config files
are — typically a scratch or work directory on the cluster, not the KAPyBARA
source directory. The fix is to put the venv's ``bin/`` on your ``PATH`` once.

**Recommended: add to** ``~/.bashrc`` **(or** ``~/.bash_profile``\ **)**

.. code-block:: bash

   echo 'export PATH="/path/to/KAPyBARA/.venv/bin:$PATH"' >> ~/.bashrc
   source ~/.bashrc

After this, ``kapybara`` works from any directory without ``uv run``:

.. code-block:: bash

   cd /scratch/my_project
   kapybara prerun -c config.yaml

**Alternative: activate the venv per session**

.. code-block:: bash

   source /path/to/KAPyBARA/.venv/bin/activate
   # kapybara is now on PATH for this shell session
   kapybara prerun -c config.yaml

**Alternative: use** ``uv run --project`` **(no PATH change needed)**

.. code-block:: bash

   uv run --project /path/to/KAPyBARA kapybara prerun -c config.yaml

This is convenient for occasional use but verbose for everyday workflows.

----

Building the documentation
---------------------------

To build these docs locally, install the optional docs dependencies and run
Sphinx:

.. code-block:: bash

   uv pip install -e ".[docs]"
   cd docs && make html

Open ``docs/_build/html/index.html`` in a browser, or serve it locally:

.. code-block:: bash

   cd docs/_build/html && python -m http.server 8080
   # Then visit http://localhost:8080
