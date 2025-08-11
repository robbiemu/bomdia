"""Additional tests for LiteLLM integration."""

import pytest
from unittest.mock import patch, MagicMock

from shared.config import config
from src.components.verbal_tag_injector.llm_based import build_llm_injector


def test_llm_injector_with_llm_spec():
    """Test that the LLM injector works when LLM_SPEC is configured."""
    # Save original config values
    original_llm_spec = config.LLM_SPEC
    original_llm_parameters = config.LLM_PARAMETERS

    try:
        # Configure the app to use an LLM
        config.LLM_SPEC = "openai/gpt-4o-mini"
        config.LLM_PARAMETERS = {"temperature": 0.7, "max_tokens": 100}

        # Create a mock response from LiteLLM
        mock_choice = MagicMock()
        mock_choice.message.content = "[S1] Hello there (laughs)"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        # Patch litellm.completion
        with patch("litellm.completion", return_value=mock_response) as mock_completion:
            # Build and run the injector
            injector = build_llm_injector()
            state = {
                "current_line": "[S1] Hello there",
                "prev_lines": [],
                "next_lines": [],
                "summary": "A greeting",
                "topic": "greeting"
            }
            result = injector(state)

            # Assertions
            assert result["modified_line"] == "[S1] Hello there (laughs)"
            mock_completion.assert_called_once()

            # Check that model spec and parameters were passed correctly
            call_args, call_kwargs = mock_completion.call_args
            assert call_kwargs["model"] == "openai/gpt-4o-mini"
            assert call_kwargs["temperature"] == 0.7
            assert call_kwargs["max_tokens"] == 100
    finally:
        # Restore original config values
        config.LLM_SPEC = original_llm_spec
        config.LLM_PARAMETERS = original_llm_parameters


def test_llm_injector_fallback_on_error():
    """Test that the LLM injector falls back to original line on error."""
    # Save original config values
    original_llm_spec = config.LLM_SPEC
    original_llm_parameters = config.LLM_PARAMETERS

    try:
        # Configure the app to use an LLM
        config.LLM_SPEC = "openai/gpt-4o-mini"
        config.LLM_PARAMETERS = {}

        # Patch litellm.completion to raise an exception
        with patch("litellm.completion", side_effect=Exception("API Error")):
            # Build and run the injector
            injector = build_llm_injector()
            state = {
                "current_line": "[S1] Hello there",
                "prev_lines": [],
                "next_lines": [],
                "summary": "A greeting",
                "topic": "greeting"
            }
            result = injector(state)

            # Should fall back to original line
            assert result["modified_line"] == "[S1] Hello there"
    finally:
        # Restore original config values
        config.LLM_SPEC = original_llm_spec
        config.LLM_PARAMETERS = original_llm_parameters
