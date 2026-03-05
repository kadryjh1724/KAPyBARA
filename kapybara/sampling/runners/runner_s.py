"""RunnerTs: T-s space TPS runner.

Manages the full relax + acquisition loop for one (T, s, replica) job.
The activity-bias field s is the active scan axis; g is fixed at ``config.g[0]``
(typically ``"0.00000"``).

All sampling logic lives in
:class:`~kapybara.sampling.runners.runner_base._RunnerBase`.
"""

from kapybara.sampling.runners.runner_base import _RunnerBase


class RunnerTs(_RunnerBase):
    """T-s space TPS runner.

    Biases the path-ensemble activity using field s in the weight
    ``exp(-s*K - g*E)``.  The g-field is fixed at the scalar value
    ``config.g[0]`` (normally ``"0.00000"``).
    """

    field_axis = "s"
