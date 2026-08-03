"""Global logging configuration for Resume Copilot.

Uses ``loguru`` to provide colorful console output and structured file
logging. The logger is configured once at import time based on
``settings.log_level``.
"""

import sys
from pathlib import Path

from loguru import logger

from resume_copilot.config import get_settings


def configure_logger() -> None:
    """Configure loguru sinks for console and file output."""
    settings = get_settings()

    # Remove any pre-configured handlers to avoid duplicate messages.
    logger.remove()

    # ------------------------------------------------------------------
    # Console sink: colored, human-readable output.
    # ------------------------------------------------------------------
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # ------------------------------------------------------------------
    # File sink: structured, rotating logs stored under outputs/logs.
    # ------------------------------------------------------------------
    log_dir = settings.outputs_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.add(
        log_dir / "resume_copilot_{time:YYYY-MM-DD}.log",
        level=settings.log_level,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
        rotation="1 day",
        retention="7 days",
        encoding="utf-8",
        enqueue=True,
    )

    logger.debug(
        "Logger configured with level='{}', data_dir='{}', outputs_dir='{}'",
        settings.log_level,
        settings.data_dir,
        settings.outputs_dir,
    )


# Configure once on module import.
configure_logger()

# Export the configured logger instance.
__all__ = ["logger"]
