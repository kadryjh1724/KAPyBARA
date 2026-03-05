"""Activity observable computation for TPS.

The activity K is a time-integrated mobility measure that counts the total
number of large per-particle displacements across consecutive frames of a
trajectory. It is used as the order parameter in the biasing weight
``exp(-s*K - g*E)``.
"""

import numpy as np


def compute_activity(pos_array: np.ndarray, box_size: float) -> int:
    """Compute the activity (number of mobile particles) from a position trajectory.

    Activity is defined as the total count of per-particle, per-frame
    squared displacements exceeding (0.3)^2, with periodic boundary
    conditions applied.

    Args:
        pos_array: Position trajectory, shape (n_frames, n_particles, 3).
        box_size: Cubic box side length for minimum-image convention.

    Returns:
        Integer activity count.
    """
    diff = pos_array[1:] - pos_array[:-1]
    diff = np.mod(diff + box_size / 2.0, box_size) - box_size / 2.0
    dist_sq = np.sum(diff * diff, axis=2)

    return int(np.sum(dist_sq > 0.3 * 0.3))
