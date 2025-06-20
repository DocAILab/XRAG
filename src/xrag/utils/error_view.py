"""
Error display module for XRAG project.

This module provides a function to display a detailed error panel.
"""

import traceback
from rich.rule import Rule
from rich.panel import Panel
from rich.align import Align
from rich.panel import Panel
from rich.align import Align
from rich.console import Console, Group


def show_error_view(error: Exception, exit_code: int = -1) -> None:
    """
    Display a detailed error panel when an uncaught exception occurs.

    Args:
        error (Exception): The exception that was caught
        exit_code (int): The exit code to use when terminating the process (default: -1)

    Returns:
        No Return
    """
    error_type = type(error).__name__
    error_message = str(error) or "[No message provided]"
    error_traceback = traceback.format_exc().strip()

    # Create a rich panel with error details
    error_panel = Panel(
        Group(
            Align.center(f"[red]{error_type}[/red]: {error_message}"),
            Rule(style="dim"),
            Align.center(f"{error_traceback}"),
            Rule(style="dim"),
            Align.center("[red]Process interrupted due to an error.[/red]"),
            Align.center(
                "If you believe this is a bug, "
                "please report this issue to the developers."
            ),
        ),
        border_style="bold red",
        title="[bold red]Exception Occurred[/bold red]",
    )

    console = Console()
    console.print(error_panel, new_line_start=True)

    exit(exit_code)
