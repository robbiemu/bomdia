"""Configuration management for the bomdia project."""

import os
from pathlib import Path
from typing import Optional

import tomli


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
        self.DIA_COMPUTE_DTYPE = os.environ.get(
            "DIA_COMPUTE_DTYPE",
            self._file_config.get("model", {}).get("dia_compute_dtype", "float16"),
        )
        # Device selection for PyTorch
        self.DIA_DEVICE = os.environ.get(
            "BOMDIA_DEVICE",
            self._file_config.get("model", {}).get("device", "auto"),
        ).lower()
        self.LLM_SPEC = os.environ.get(
            "LLM_SPEC",
            self._file_config.get("model", {}).get("llm_spec", None),
        )
        # Load model parameters, defaulting to an empty dict
        self.LLM_PARAMETERS = self._file_config.get("model", {}).get("parameters", {})
        self.DIA_GENERATE_PARAMS = self._file_config.get("model", {}).get(
            "dia_generate_params", {}
        )

        # Audio settings
        self.AUDIO_OUTPUT_FORMAT = os.environ.get(
            "AUDIO_OUTPUT_FORMAT",
            self._file_config.get("audio", {}).get("output_format", "mp3"),
        )
        self.AUDIO_SAMPLING_RATE = int(
            os.environ.get(
                "AUDIO_SAMPLING_RATE",
                self._file_config.get("audio", {}).get("sampling_rate", "44100"),
            )
        )
        self.AUDIO_SAMPLE_WIDTH = int(
            os.environ.get(
                "AUDIO_SAMPLE_WIDTH",
                self._file_config.get("audio", {}).get("sample_width", "2"),
            )
        )
        self.AUDIO_CHANNELS = int(
            os.environ.get(
                "AUDIO_CHANNELS",
                self._file_config.get("audio", {}).get("channels", "1"),
            )
        )

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

        seed = os.environ.get(
            "SEED",
            self._file_config.get("pipeline", {}).get("seed", None),
        )
        if seed is not None:
            self.SEED: Optional[int] = int(seed)
        else:
            self.SEED = None

        self.FULLY_DETERMINISTIC = os.environ.get(
            "FULLY_DETERMINISTIC",
            self._file_config.get("pipeline", {}).get("fully_deterministic", False),
        )

        self.MIN_CHUNK_DURATION = os.environ.get(
            "MIN_CHUNK_DURATION",
            self._file_config.get("pipeline", {}).get("min_chunk_duration", 5.0),
        )

        self.MAX_CHUNK_DURATION = os.environ.get(
            "MAX_CHUNK_DURATION",
            self._file_config.get("pipeline", {}).get("max_chunk_duration", 10.0),
        )

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

        # Synthetic prompt generation settings
        self.GENERATE_SYNTHETIC_PROMPTS = (
            os.environ.get(
                "GENERATE_SYNTHETIC_PROMPTS",
                str(
                    self._file_config.get("pipeline", {}).get(
                        "generate_synthetic_prompts", True
                    )
                ),
            ).lower()
            == "true"
        )

        self.GENERATE_PROMPT_OUTPUT_DIR = os.environ.get(
            "GENERATE_PROMPT_OUTPUT_DIR",
            self._file_config.get("generate_prompt", {}).get(
                "output_dir", "synthetic_prompts"
            ),
        )

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
