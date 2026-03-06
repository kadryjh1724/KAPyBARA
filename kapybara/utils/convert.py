"""Data-type conversion utilities for LAMMPS and KAPyBARA.

Provides functions for converting between NumPy arrays and ctypes pointers
(used by LAMMPS scatter_atoms / gather_atoms calls), NumPy arrays and
formatted string lists (used for field parameter handling), and elapsed-time
formatting.
"""

import numpy as np
from typing import List, Any
from ctypes import POINTER, c_int, c_double


def npy2Cint(arr: np.ndarray) -> Any:
    """Convert a NumPy array to a ctypes int32 pointer for LAMMPS.

    Args:
        arr: Input array; will be cast to int32.

    Returns:
        A ctypes POINTER(c_int) pointing to the array data.
    """
    arr = np.asarray(arr, dtype=np.int32)
    return arr.ctypes.data_as(POINTER(c_int))


def npy2Cdouble(arr: np.ndarray) -> Any:
    """Convert a NumPy array to a ctypes float64 pointer for LAMMPS.

    Args:
        arr: Input array; will be cast to float64.

    Returns:
        A ctypes POINTER(c_double) pointing to the array data.
    """
    arr = np.asarray(arr, dtype=np.float64)
    return arr.ctypes.data_as(POINTER(c_double))


def str2npy(strList: List[str]) -> np.ndarray:
    """Convert a list of numeric strings to a float64 NumPy array.

    Args:
        strList: List of strings representing floating-point values.

    Returns:
        1-D NumPy array of float64 values.
    """
    return np.array([float(s) for s in strList])


def npy2str(arr: np.ndarray, n: int) -> List[str]:
    """Convert a NumPy array to a list of fixed-decimal formatted strings.

    Rounds each element to ``n`` decimal places and formats it as a string
    of the form ``"{value:.nf}"``. Scalar or single-element arrays return
    a one-element list.

    Args:
        arr: Input NumPy array (scalar or 1-D).
        n: Number of decimal places.

    Returns:
        List of formatted strings, one per element.
    """
    rounded_arr = np.round(arr, decimals=n)
    if len(arr.shape) == 0 or (len(arr.shape) == 1 and arr.shape[0] == 1):
        return [f"{float(rounded_arr):.{n}f}"]
    return [f"{a:.{n}f}" for a in rounded_arr]


def strfdelta_short(delta_t: float) -> str:
    """Format elapsed seconds as ``DDd HHh MMm`` (zero-padded, no seconds).

    Same decomposition as :func:`strfdelta2` but drops the seconds field.
    Output is exactly 11 characters.

    Args:
        delta_t: Elapsed time in seconds.

    Returns:
        Formatted string, e.g. ``"01d 02h 48m"``.
    """
    days    = int(delta_t // 86400)
    remain  = delta_t % 86400
    hours   = int(remain // 3600)
    minutes = int(remain % 3600 // 60)
    return f"{days:02d}d {hours:02d}h {minutes:02d}m"


def strfdelta2(delta_t: float) -> str:
    """Format elapsed seconds as DDd HHh MMm SSs (zero-padded)."""
    days    = int(delta_t // (24 * 3600))
    remain  = delta_t % (24 * 3600)
    hours   = int(remain // 3600)
    remain  = remain % 3600
    minutes = int(remain // 60)
    seconds = int(remain % 60)
    return f"{days:02d}d {hours:02d}h {minutes:02d}m {seconds:02d}s"