import os
from unittest.mock import patch

import pytest
from shared.config import Config as AppConfig


@pytest.fixture
def mock_toml_load():
    with patch("tomli.load") as mock_load:
        yield mock_load


def test_config_loading(mocker):
    mocker.patch(
        "tomli.load",
        return_value={
            "model": {"llm_spec": "some_model", "parameters": {"temperature": 0.5}},
            "pipeline": {"max_tag_rate": 0.5},
            "director_agent": {
                "global_summary_prompt": "global",
                "unified_moment_analysis_prompt": "unified",
                "quota_exceeded_note": "quota",
            },
            "actor_agent": {"task_directive_template": "task"},
        },
    )

    config = AppConfig()

    assert config.LLM_SPEC == "some_model"
    assert config.LLM_PARAMETERS == {"temperature": 0.5}
    assert config.MAX_TAG_RATE == 0.5
    assert config.director_agent["global_summary_prompt"] == "global"
    assert config.director_agent["unified_moment_analysis_prompt"] == "unified"
    assert config.director_agent["quota_exceeded_note"] == "quota"
    assert config.actor_agent["task_directive_template"] == "task"


def test_config_env_override(mock_toml_load):
    with patch.dict(os.environ, {"LLM_SPEC": "env_model", "MAX_TAG_RATE": "0.9"}):
        mock_toml_load.return_value = {
            "llm": {"spec": "some_model", "parameters": {"temperature": 0.5}},
            "max_tag_rate": 0.5,
            "director_agent": {},
            "actor_agent": {},
        }

        config = AppConfig()

        assert config.LLM_SPEC == "env_model"
        assert config.MAX_TAG_RATE == 0.9
