"""Internal validation error type for configuration parsing.

Used to carry an underlying exception and two message strings through the
validation pipeline so that prettyError can render them with full context.
"""


class _ValidationError(Exception):
    """Wraps a base exception with two human-readable message strings.

    Attributes:
        exc: The underlying exception instance (e.g., ``ValueError()``).
        message1: Primary error description.
        message2: Optional secondary detail or current-value context.
    """

    def __init__(self, exc, message1, message2=None):
        """Initialise with an exception and message strings.

        Args:
            exc: The underlying exception (e.g., ``ValueError()``).
            message1: Primary error description.
            message2: Optional detail string shown after a rule separator.
        """
        self.exc = exc
        self.message1 = message1
        self.message2 = message2