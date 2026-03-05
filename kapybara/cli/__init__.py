"""Command-line interface sub-package.

Exports the main() entry point, which dispatches to the prerun, run,
monitor, queue, and analysis sub-commands via argparse.
"""

from kapybara.cli.cli import main
