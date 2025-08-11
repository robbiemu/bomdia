"""Tests for the verbal tag injector component."""

from unittest.mock import MagicMock

from shared.config import config
from src.components.verbal_tag_injector.llm_based import build_llm_injector


def test_prompt_loading():
    """Test that prompts are correctly loaded from the config."""
    # Check that the prompts are loaded correctly
    assert config.VERBAL_TAG_INJECTOR_SYSTEM_PROMPT is not None
    assert config.VERBAL_TAG_INJECTOR_HUMAN_PROMPT_TEMPLATE is not None

    # Check that the prompts contain expected content
    assert "concise transcript editor" in config.VERBAL_TAG_INJECTOR_SYSTEM_PROMPT
    assert "Prev lines:" in config.VERBAL_TAG_INJECTOR_HUMAN_PROMPT_TEMPLATE


def test_llm_injector_with_litellm_mock(mocker):
    """Test that the injector calls litellm.completion and processes the response."""
    # 1. Configure the app to use an LLM
    config.LLM_SPEC = "ollama/test-model"
    config.LLM_PARAMETERS = {"temperature": 0.1}

    # 2. Create a mock response from LiteLLM
    mock_choice = MagicMock()
    mock_choice.message.content = "[S1] (chuckle) This is a test."
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    # 3. Patch litellm.completion
    mock_completion = mocker.patch("litellm.completion", return_value=mock_response)

    # 4. Build and run the injector
    injector = build_llm_injector()
    state = {"current_line": "[S1] This is a test."} # simplified state for test
    result = injector(state)

    # 5. Assertions
    assert result["modified_line"] == "[S1] (chuckle) This is a test."
    mock_completion.assert_called_once()
    # Check that model spec and parameters were passed correctly
    call_args, call_kwargs = mock_completion.call_args
    assert call_kwargs["model"] == "ollama/test-model"
    assert call_kwargs["temperature"] == 0.1
