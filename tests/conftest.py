"""Test configuration and fixtures for the bomdia project."""

import os


def pytest_configure(config):
    """Configure pytest."""
    # Use in-memory database for testing by default, unless specifically overridden
    if "REHEARSAL_CHECKPOINT_PATH" not in os.environ:
        os.environ["REHEARSAL_CHECKPOINT_PATH"] = ":memory:"
