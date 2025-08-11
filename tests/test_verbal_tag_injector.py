"""Tests for the verbal tag injector component."""

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


def test_llm_injector_creation():
    """Test that we can create an LLM injector."""
    # Create a mock LLM
    class MockLLM:
        def invoke(self, messages):
            # Return a mock response
            class MockResponse:
                content = "[S1] Hello there (laughs)"
            return MockResponse()

    # Build the injector
    llm = MockLLM()
    injector = build_llm_injector(llm)

    # Check that the injector is created correctly
    assert injector is not None
    assert callable(injector)
