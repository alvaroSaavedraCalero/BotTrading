"""Utility helpers for project wide logging configuration."""

import logging
import sys
from typing import Optional


def setup_logging(level: int = logging.INFO, log_file: Optional[str] = None) -> None:
    """Configure root logger with console and optional file handlers.

    Parameters
    ----------
    level:
        Logging level for the root logger. Defaults to ``logging.INFO``.
    log_file:
        Optional path to a file where logs will also be written.
    """
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True,
    )

