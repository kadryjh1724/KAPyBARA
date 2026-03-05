"""RunnerSg: doubly-biased s-g space TPS runner (not yet implemented).

Placeholder for future doubly-biased sampling in (s, g) space, where both
the activity K and the energy E are biased simultaneously.
"""


class RunnerSg:
    """Doubly-biased s-g space TPS runner (stub).

    Not yet implemented. Attempting to use this runner raises
    :exc:`NotImplementedError`.
    """

    def run(self, T: str, field_value: str, replica_index: int) -> None:
        """Run TPS for one (T, s, g, replica) point.

        Args:
            T: Temperature string.
            field_value: Combined field value string.
            replica_index: Zero-based replica index.

        Raises:
            NotImplementedError: Always, as this runner is not yet implemented.
        """
        raise NotImplementedError("s-g space TPS runner is not implemented yet.")
