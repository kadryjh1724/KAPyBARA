"""Configuration sub-package.

Exports the frozen SimulationConfig dataclass, the load_config loader, and
PathManager for all simulation directory paths.
"""

from kapybara.config.schema import SimulationConfig
from kapybara.config.loader import load_config
from kapybara.config.paths import PathManager
