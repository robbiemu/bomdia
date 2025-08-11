"""Configuration management for the bomdia project."""

import os
from pathlib import Path

import tomli  # For reading TOML files


class Config:
    """Centralized configuration management from config.toml with env override."""

    def __init__(self) -> None:
        # Load main configuration from file
        config_path = Path("config/app.toml")
        if config_path.exists():
            with open(config_path, "rb") as f:
                file_config = tomli.load(f)
        else:
            file_config = {}

        # Load prompts configuration from file
        prompts_path = Path("config/prompts.toml")
        if prompts_path.exists():
            with open(prompts_path, "rb") as f:
                prompts_config = tomli.load(f)
        else:
            prompts_config = {}

        # Model checkpoints and API settings
        self.DIA_CHECKPOINT = os.environ.get(
            "DIA_CHECKPOINT",
            file_config.get("model", {}).get("dia_checkpoint", "nari-labs/Dia-1.6B"),
        )
        self.DIA_CHECKPOINT_REVISION = os.environ.get(
            "DIA_CHECKPOINT_REVISION",
            file_config.get("model", {}).get("dia_checkpoint_revision", "main"),
        )
        self.LLM_SPEC = os.environ.get(
            "LLM_SPEC",
            file_config.get("model", {}).get("llm_spec", None),
        )
        # Load model parameters, defaulting to an empty dict
        self.LLM_PARAMETERS = file_config.get("model", {}).get("parameters", {})

        # Pipeline behavior constants
        self.CONTEXT_WINDOW = int(
            os.environ.get(
                "CONTEXT_WINDOW",
                file_config.get("pipeline", {}).get("context_window", "2"),
            )
        )
        self.PAUSE_PLACEHOLDER = os.environ.get(
            "PAUSE_PLACEHOLDER",
            file_config.get("pipeline", {}).get(
                "pause_placeholder", "[insert-verbal-tag-for-pause]"
            ),
        )
        self.MAX_TAG_RATE = float(
            os.environ.get(
                "MAX_TAG_RATE",
                file_config.get("pipeline", {}).get("max_tag_rate", "0.15"),
            )
        )
        self.AVG_WPS = float(
            os.environ.get(
                "AVG_WPS", file_config.get("pipeline", {}).get("avg_wps", "2.5")
            )
        )
        self.MAX_NEW_TOKENS_CAP = int(
            os.environ.get(
                "MAX_NEW_TOKENS_CAP",
                file_config.get("pipeline", {}).get("max_new_tokens_cap", "1600"),
            )
        )

        seed = os.environ.get(
            "SEED",
            file_config.get("pipeline", {}).get("seed", None),
        )
        if seed is not None:
            self.SEED = int(seed)

        # Verbal tags and line combiners (from file only, as these are lists)
        self.VERBAL_TAGS = file_config.get("tags", {}).get(
            "verbal_tags",
            [
                "(laughs)",
                "(clears throat)",
                "(sighs)",
                "(gasps)",
                "(coughs)",
                "(singing)",
                "(sings)",
                "(mumbles)",
                "(beep)",
                "(groans)",
                "(sniffs)",
                "(claps)",
                "(screams)",
                "(inhales)",
                "(exhales)",
                "(applause)",
                "(burps)",
                "(humming)",
                "(sneezes)",
                "(chuckle)",
                "(whistles)",
            ],
        )

        self.LINE_COMBINERS = file_config.get("tags", {}).get(
            "line_combiners",
            [
                "…um,",
                "- uh -",
                "— hmm —",
            ],
        )

        # Prompts
        self.director_agent = prompts_config.get("director_agent", {})
        self.actor_agent = prompts_config.get("actor_agent", {})


# Global configuration instance
config = Config()
