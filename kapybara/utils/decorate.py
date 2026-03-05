"""Timing decorator for benchmarking simulation phases."""

import time
import functools


def measureTime(func):
    """Decorator that measures and returns the wall-clock execution time.

    Wraps the decorated function so that it returns a two-element tuple
    ``(result, elapsed_seconds)`` instead of just ``result``.

    Args:
        func: The function to time.

    Returns:
        A wrapped function with the same signature as ``func`` that returns
        ``(original_return_value, elapsed_seconds)``.

    Example::

        @measureTime
        def run_md():
            ...

        result, dt = run_md()
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        return result, execution_time
    
    return wrapper