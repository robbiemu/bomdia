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
                self._file_config = tomli.load(f)
        else:
            self._file_config = {}

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
            self._file_config.get("model", {}).get(
                "dia_checkpoint", "nari-labs/Dia-1.6B"
            ),
        )
        self.DIA_CHECKPOINT_REVISION = os.environ.get(
            "DIA_CHECKPOINT_REVISION",
            self._file_config.get("model", {}).get("dia_checkpoint_revision", "main"),
        )
        self.LLM_SPEC = os.environ.get(
            "LLM_SPEC",
            self._file_config.get("model", {}).get("llm_spec", None),
        )
        # Load model parameters, defaulting to an empty dict
        self.LLM_PARAMETERS = self._file_config.get("model", {}).get("parameters", {})

        # Pipeline behavior constants
        self.CONTEXT_WINDOW = int(
            os.environ.get(
                "CONTEXT_WINDOW",
                self._file_config.get("pipeline", {}).get("context_window", "2"),
            )
        )
        self.PAUSE_PLACEHOLDER = os.environ.get(
            "PAUSE_PLACEHOLDER",
            self._file_config.get("pipeline", {}).get(
                "pause_placeholder", "[insert-verbal-tag-for-pause]"
            ),
        )
        self.MAX_TAG_RATE = float(
            os.environ.get(
                "MAX_TAG_RATE",
                self._file_config.get("pipeline", {}).get("max_tag_rate", "0.15"),
            )
        )
        self.AVG_WPS = float(
            os.environ.get(
                "AVG_WPS", self._file_config.get("pipeline", {}).get("avg_wps", "2.5")
            )
        )
        self.MAX_NEW_TOKENS_CAP = int(
            os.environ.get(
                "MAX_NEW_TOKENS_CAP",
                self._file_config.get("pipeline", {}).get("max_new_tokens_cap", "1600"),
            )
        )

        seed = os.environ.get(
            "SEED",
            self._file_config.get("pipeline", {}).get("seed", None),
        )
        if seed is not None:
            self.SEED = int(seed)

        # Verbal tags and line combiners (from file only, as these are lists)
        self.VERBAL_TAGS = self._file_config.get("tags", {}).get(
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

        self.LINE_COMBINERS = self._file_config.get("tags", {}).get(
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

        # Add rate control settings from app config to director_agent
        rate_control = self._file_config.get("director_agent", {}).get(
            "rate_control", {}
        )
        if rate_control:
            self.director_agent["rate_control"] = rate_control

        # Add review settings from app config to director_agent
        review_cfg = self._file_config.get("director_agent", {}).get("review", {})
        review_mode = os.environ.get(
            "DIRECTOR_AGENT_REVIEW_MODE", review_cfg.get("mode", "procedural")
        )
        if "review" not in self.director_agent:
            self.director_agent["review"] = {}
        self.director_agent["review"]["mode"] = review_mode

    @property
    def REHEARSAL_CHECKPOINT_PATH(self) -> str:
        """Get the rehearsal checkpoint path."""
        return os.environ.get(
            "REHEARSAL_CHECKPOINT_PATH",
            self._file_config.get("persistence", {}).get(
                "rehearsal_checkpoint_path", "rehearsal_checkpoints.sqlite"
            ),
        )


# Global configuration instance
config = Config()
