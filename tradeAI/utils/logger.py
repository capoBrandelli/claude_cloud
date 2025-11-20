"""
Logging utilities for TradeAI
"""

import sys
from pathlib import Path
from typing import Optional

from loguru import logger


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    rotation: str = "100 MB",
    retention: str = "30 days",
    compression: str = "zip",
    format_string: Optional[str] = None,
) -> None:
    """
    Set up logging configuration for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (None to disable file logging)
        rotation: When to rotate log files
        retention: How long to keep old log files
        compression: Compression format for rotated logs
        format_string: Custom format string for log messages
    """
    # Remove default logger
    logger.remove()

    # Default format
    if format_string is None:
        format_string = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )

    # Add console logger
    logger.add(
        sys.stderr,
        format=format_string,
        level=log_level,
        colorize=True,
    )

    # Add file logger if specified
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            log_file,
            format=format_string,
            level=log_level,
            rotation=rotation,
            retention=retention,
            compression=compression,
            enqueue=True,  # Thread-safe logging
        )

    logger.info(f"Logging initialized with level: {log_level}")


def get_logger(name: str):
    """
    Get a logger instance with the specified name.

    Args:
        name: Logger name (typically __name__ of the module)

    Returns:
        Logger instance
    """
    return logger.bind(name=name)


# Default logger instance
default_logger = logger
