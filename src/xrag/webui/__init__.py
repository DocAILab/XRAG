"""XRAG WebUI package.

Public entry point is :func:`run_web_ui`, exposed for ``xrag-cli webui``.
"""

from .launcher import run_web_ui

__all__ = ["run_web_ui"]