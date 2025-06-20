"""
Logger utility for XRAG project.

This module provides a centralized logging configuration that can be used
across all XRAG modules for consistent log formatting and handling.
"""

import sys
import logging
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.logging import RichHandler


def setup_logger(
    name: str = "xrag",
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
    use_config: bool = True,
    use_rich: bool = True,
) -> logging.Logger:
    """
    Set up a logger with consistent formatting and handlers.

    Args:
        name (str): Logger name (default: "xrag")
        level (str, optional): Logging level. If None and use_config=True, reads from config
        log_file (str, optional): Path to log file. If None and use_config=True, reads from config
        format_string (str, optional): Custom format string. If None and use_config=True, reads from config
        use_config (bool): Whether to use configuration values as defaults (default: True)
        use_rich (bool): Whether to use Rich console handler for colored output (default: True)

    Returns:
        Configured logger instance
    """
    if use_config:
        try:
            from ..config import Config

            cfg = Config()

            if level is None and hasattr(cfg, "log_level"):
                level = cfg.log_level
            if log_file is None and hasattr(cfg, "log_file"):
                log_file = cfg.log_file
            if format_string is None and hasattr(cfg, "log_format"):
                format_string = cfg.log_format
        except Exception:
            # If config loading fails, use defaults
            pass

    # Set defaults if still None
    if level is None:
        level = "INFO"
    if format_string is None:
        format_string = "[%(asctime)s,%(msecs)03d] %(levelname)-8s - (%(name)s) %(filename)s, ln %(lineno)d: %(message)s"

    logger = logging.getLogger(name)
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    logger.handlers.clear()

    # Console handler with Rich formatting
    if use_rich:
        console = Console()
        console_handler = RichHandler(
            console=console,
            show_time=True,
            show_level=True,
            show_path=False,
            omit_repeated_times=False,
            log_time_format="[%H:%M:%S]",
        )
        rich_formatter = logging.Formatter(
            "(%(name)s) %(filename)s, ln %(lineno)d: %(message)s"
        )
        console_handler.setFormatter(rich_formatter)
        console_handler.setLevel(numeric_level)
        logger.addHandler(console_handler)
    else:
        # Fallback to standard console handler
        formatter = logging.Formatter(format_string, datefmt="%H:%M:%S")
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler - only create if log_file is not empty
    if log_file and isinstance(log_file, str) and log_file.strip():
        log_path = Path(log_file).resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_formatter = logging.Formatter(format_string, datefmt="%H:%M:%S")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def get_logger(name: str = "xrag") -> logging.Logger:
    """
    Get or create a logger instance.

    Args:
        name (str): Logger name (default: "xrag")

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)

    # If logger has no handlers, set it up with default configuration
    if not logger.handlers:
        return setup_logger(name)

    return logger


default_logger = setup_logger("xrag")


def get_module_logger(module_name: str) -> logging.Logger:
    """
    Get a logger for a specific module.

    Args:
        module_name (str): Name of the module (e.g., __name__)

    Returns:
        Logger instance for the module
    """
    # Extract the module name relative to xrag package
    if "xrag" in module_name:
        parts = module_name.split('.')
        if 'xrag' in parts:
            xrag_index = parts.index('xrag')
            logger_name = '.'.join(parts[xrag_index:])
        else:
            logger_name = module_name
    else:
        logger_name = f"xrag.{module_name}"

    return get_logger(logger_name)
