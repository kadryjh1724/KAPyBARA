"""RunnerTg: T-g space TPS runner.

Manages the full relax + acquisition loop for one (T, g, replica) job.
The energy-bias field g is the active scan axis; s is fixed at ``config.s[0]``
(typically ``"0.00000"``).

All sampling logic lives in
:class:`~kapybara.sampling.runners.runner_base._RunnerBase`.
"""

from kapybara.sampling.runners.runner_base import _RunnerBase


class RunnerTg(_RunnerBase):
    """T-g space TPS runner.

    Biases the path-ensemble energy using field g in the weight
    ``exp(-s*K - g*E)``.  The s-field is fixed at the scalar value
    ``config.s[0]`` (normally ``"0.00000"``).
    """

    field_axis = "g"
