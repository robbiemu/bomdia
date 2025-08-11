"""Tests for the configuration system."""

from shared.config import Config


def test_config_defaults():
    """Test that config loads defaults correctly."""
    # Create a config instance
    config = Config()

    # Test default values
    assert config.DIA_CHECKPOINT == "nari-labs/Dia-1.6B-0626"
    assert config.LLM_SPEC == "openai/gpt-4o-mini"
    assert config.CONTEXT_WINDOW == 2
    assert config.PAUSE_PLACEHOLDER == "[insert-verbal-tag-for-pause]"
    assert config.MAX_TAG_RATE == 0.15
    assert config.AVG_WPS == 2.5
    assert config.MAX_NEW_TOKENS_CAP == 1600

    # Test list values
    assert "(laughs)" in config.VERBAL_TAGS
    assert "…um," in config.LINE_COMBINERS

    # Test LLM parameters
    assert config.LLM_PARAMETERS == {"temperature": 0.5, "max_tokens": 4096}


def test_config_env_override(monkeypatch):
    """Test that environment variables override config values."""
    # Set environment variables
    monkeypatch.setenv("DIA_CHECKPOINT", "test/checkpoint")
    monkeypatch.setenv("CONTEXT_WINDOW", "5")
    monkeypatch.setenv("MAX_TAG_RATE", "0.3")
    monkeypatch.setenv("LLM_SPEC", "ollama/test-model")

    # Create a new config instance
    config = Config()

    # Test overridden values
    assert config.DIA_CHECKPOINT == "test/checkpoint"
    assert config.CONTEXT_WINDOW == 5
    assert config.MAX_TAG_RATE == 0.3
    assert config.LLM_SPEC == "ollama/test-model"


def test_config_toml_override(tmp_path, monkeypatch):
    """Test that TOML config file values are loaded correctly."""
    config_path = tmp_path / "app.toml"

    # Create a temporary TOML config file with custom values
    test_config_content = """
[model]
dia_checkpoint = "test/toml/checkpoint"
llm_spec = "test:model"
dia_checkpoint_revision = "test-revision"

[model.parameters]
temperature = 0.7
max_tokens = 200

[pipeline]
context_window = 3
pause_placeholder = "[test-placeholder]"
max_tag_rate = 0.25
avg_wps = 3.0
max_new_tokens_cap = 2000
"""

    # Write the test config
    config_path.write_text(test_config_content)

    # Mock the config file path
    monkeypatch.setattr(
        "shared.config.Path",
        lambda x: config_path if x == "config/app.toml" else tmp_path / x,
    )

    # Create a new config instance
    config = Config()

    # Test TOML values
    assert config.DIA_CHECKPOINT == "test/toml/checkpoint"
    assert config.LLM_SPEC == "test:model"
    assert config.CONTEXT_WINDOW == 3
    assert config.PAUSE_PLACEHOLDER == "[test-placeholder]"
    assert config.MAX_TAG_RATE == 0.25
    assert config.AVG_WPS == 3.0
    assert config.MAX_NEW_TOKENS_CAP == 2000
    assert config.DIA_CHECKPOINT_REVISION == "test-revision"
    assert config.LLM_PARAMETERS == {"temperature": 0.7, "max_tokens": 200}
