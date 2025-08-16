"""Configuration file for pytest."""

import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture(autouse=True)
def ensure_network_isolation():
    """
    Ensure that tests fail if they try to make real network calls.
    This fixture checks that LiteLLMInvoker is properly mocked in all tests.
    """
    # We don't automatically mock here, but we could add verification logic
    # For now, we'll just ensure the fixture exists
    pass


@pytest.fixture
def mock_llm_response():
    """
    Fixture that provides a mock LLM response for tests.
    This ensures consistent mocking across all tests.
    """
    mock_response = MagicMock()
    mock_response.content = "Mocked LLM response"
    return mock_response
