"""TPS runner implementations.

Exports RunnerTg (T-g space) and RunnerTs (T-s space), both backed by the
shared :class:`~kapybara.sampling.runners.runner_base._RunnerBase` logic.
RunnerSg (s-g doubly-biased space) is a stub pending future implementation.
"""

from kapybara.sampling.runners.runner_g import RunnerTg
from kapybara.sampling.runners.runner_s import RunnerTs
from kapybara.sampling.runners.runner_sg import RunnerSg
