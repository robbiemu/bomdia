"""Tests for the configuration system."""

from shared.config import Config


def test_config_defaults():
    """Test that config loads defaults correctly."""
    # Create a config instance
    config = Config()

    # Test default values
    assert config.DIA_CHECKPOINT == "nari-labs/Dia-1.6B"
    assert config.OPENAI_MODEL_NAME == "openai:gpt-4o-mini"
    assert config.CONTEXT_WINDOW == 2
    assert config.PAUSE_PLACEHOLDER == "[insert-verbal-tag-for-pause]"
    assert config.MAX_TAG_RATE == 0.15
    assert config.AVG_WPS == 2.5
    assert config.MAX_NEW_TOKENS_CAP == 1600

    # Test list values
    assert "(laughs)" in config.VERBAL_TAGS
    assert "…um," in config.LINE_COMBINERS


def test_config_env_override(monkeypatch):
    """Test that environment variables override config values."""
    # Set environment variables
    monkeypatch.setenv("DIA_CHECKPOINT", "test/checkpoint")
    monkeypatch.setenv("CONTEXT_WINDOW", "5")
    monkeypatch.setenv("MAX_TAG_RATE", "0.3")

    # Create a new config instance
    config = Config()

    # Test overridden values
    assert config.DIA_CHECKPOINT == "test/checkpoint"
    assert config.CONTEXT_WINDOW == 5
    assert config.MAX_TAG_RATE == 0.3


def test_config_toml_override(tmp_path, monkeypatch):
    """Test that TOML config file values are loaded correctly."""
    config_path = tmp_path / "app.toml"

    # Create a temporary TOML config file with custom values
    test_config_content = """
[model]
dia_checkpoint = "test/toml/checkpoint"
openai_model = "test:model"
dia_checkpoint_revision = "test-revision"

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
    assert config.OPENAI_MODEL_NAME == "test:model"
    assert config.CONTEXT_WINDOW == 3
    assert config.PAUSE_PLACEHOLDER == "[test-placeholder]"
    assert config.MAX_TAG_RATE == 0.25
    assert config.AVG_WPS == 3.0
    assert config.MAX_NEW_TOKENS_CAP == 2000
    assert config.DIA_CHECKPOINT_REVISION == "test-revision"
