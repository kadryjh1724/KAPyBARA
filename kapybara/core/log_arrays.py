"""Log-array initialisation for MD trajectory storage.

Provides a factory for the zeroed dictionary used to collect per-frame
positions, velocities, and energies during LAMMPS MD runs. The arrays are
pre-allocated to avoid dynamic resizing in the inner MD loop.
"""

import numpy as np
from typing import Dict


def initialize_log_dict(nloops: int, n_particles: int) -> Dict[str, np.ndarray]:
    """Create a zeroed log dictionary for storing trajectory data.

    Returns a dict with keys:
        "pos": shape (nloops + 1, n_particles, 3)
        "vel": shape (nloops + 1, n_particles, 3)
        "pe":  shape (nloops + 1,)
        "ke":  shape (nloops + 1,)
    """
    return {
        "pos": np.zeros((nloops + 1, n_particles, 3)),
        "vel": np.zeros((nloops + 1, n_particles, 3)),
        "pe": np.zeros(nloops + 1),
        "ke": np.zeros(nloops + 1),
    }
