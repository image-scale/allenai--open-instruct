"""Logging utilities for consistent formatting across the project."""

import logging


def setup_logger(name: str | None = None, rank: int = 0) -> logging.Logger:
    """Set up a logger with consistent formatting across the project.

    This function configures logging.basicConfig with a standard format
    that includes timestamp, level, filename, line number, and message.
    It only configures basicConfig once to avoid overwriting existing config.

    Args:
        name: Logger name (typically __name__). If None, returns root logger.
        rank: Process rank in distributed training. Only rank 0 logs INFO.

    Returns:
        Logger instance with the specified name
    """
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO if rank == 0 else logging.WARNING,
            format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    return logging.getLogger(name)
