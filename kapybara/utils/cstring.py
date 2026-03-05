"""Rich-based console output helpers for notifications, warnings, and errors.

All three functions use ``sys._getframe(1)`` to embed the caller's qualified
name in the panel title, providing automatic source-location attribution in
log output.
"""

import sys
from rich import box
from rich.rule import Rule
from rich.panel import Panel
from rich.console import Console, Group


def prettyNotification(console: Console, message1: str, message2: str = None) -> None:
    """Print a formatted notification panel to stdout.

    Args:
        console: Rich Console instance for rendering.
        message1: Primary notification text (top of panel body).
        message2: Optional detail text (shown below a rule separator).
    """
    frame = sys._getframe(1)
    funcName = f"{frame.f_globals.get("__name__", "")}.{frame.f_code.co_qualname}"
    title = f"[bold navy_blue]KAPyBARA[/bold navy_blue] "
    title += f"[bold bright_yellow]Notification[/bold bright_yellow] :bell: "
    title += f"[default]in [underline]{funcName}[/underline][/default]"

    # Double-row notification
    if message2 is not None:
        console.print(
            Panel(
                Group(message1, Rule(style="dim"), ">>> " + message2),
                box=box.SQUARE, title=title, title_align="left",
                expand=False, border_style="bright_yellow"
            )
        )
    # Single-row notification
    else:
        console.print(
            Panel(
                message1,
                box=box.SQUARE, title=title, title_align="left",
                expand=False, border_style="bright_yellow"
            )
        )

def prettyWarning(console_stderr: Console, message1: str, message2: str) -> None:
    """Print a formatted warning panel to stderr.

    Always renders two lines: ``message1`` above a rule, ``message2`` below.

    Args:
        console_stderr: Rich Console instance targeting stderr.
        message1: Primary warning description.
        message2: Detail or current-value context.
    """
    frame = sys._getframe(1)
    funcName = f"{frame.f_globals.get("__name__", "")}.{frame.f_code.co_qualname}"
    title = f"[bold navy_blue]KAPyBARA[/bold navy_blue] "
    title += f"[bold orange_red1]Warning[/bold orange_red1] :exclamation: "
    title += f"[default]in [underline]{funcName}[/underline][/default]"

    # Warning is always double-rowv
    console_stderr.print(
        Panel(
            Group(message1, Rule(style="dim"), ">>> " + message2),
            box=box.SQUARE, title=title, title_align="left",
            expand=False, border_style="orange_red1"
        )
    )

def prettyError(console_stderr: Console, exc: Exception,
                message1: str, message2: str = None) -> None:
    """Print a formatted error panel to stderr.

    The panel title includes the exception type name and the caller's
    qualified function name.

    Args:
        console_stderr: Rich Console instance targeting stderr.
        exc: The exception whose type name is shown in the panel title.
        message1: Primary error description.
        message2: Optional detail text (shown below a rule separator).
    """
    frame = sys._getframe(1)
    errorName = type(exc).__name__
    funcName = f"{frame.f_globals.get("__name__", "")}.{frame.f_code.co_qualname}"
    title = f"[bold navy_blue]KAPyBARA[/bold navy_blue] "
    title += f"[bold red3]{errorName}[/bold red3] :x: "
    title += f"[default]in [underline]{funcName}[/underline][/default]"

    # Double-row error
    if message2 is not None:
        console_stderr.print(
            Panel(
                Group(message1, Rule(style="dim"), ">>> " + message2),
                box=box.SQUARE, title=title, title_align="left",
                expand=False, border_style="red3"
            )
        )
    # Single-row error
    else:
        console_stderr.print(
            Panel(
                message1,
                box=box.SQUARE, title=title, title_align="left",
                expand=False, border_style="red3"
            )
        )