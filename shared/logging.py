"""Centralized logging configuration for the bomdia application."""

import logging
import sys


def setup_logger(level: int = logging.WARNING) -> None:
    """
    Configure the root logger for the application and
    suppress noisy third-party loggers.

    Args:
        level: The logging level to set for the root logger.
    """
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)

    # Ensure we don't have duplicate handlers
    root_logger.propagate = False

    # Silence noisy third-party loggers
    noisy_loggers = [
        "litellm",
        "LiteLLM",  # Also try this variant
        "openai",
        "httpx",
        "httpcore",
        "urllib3",
        "asyncio",
    ]

    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.

    Args:
        name: The name for the logger (typically __name__).

    Returns:
        A configured logger instance.
    """
    return logging.getLogger(name)
