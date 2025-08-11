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
        self.OPENAI_MODEL_NAME = os.environ.get(
            "OPENAI_MODEL",
            file_config.get("model", {}).get("openai_model", "openai:gpt-4o-mini"),
        )

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
        default_system_prompt = (
            "You are a concise transcript editor. You receive a single transcript line "
            "prefixed by a speaker tag ([S1] or [S2]) and the surrounding context. "
            "Return ONLY the updated single line (no commentary). Rules:\n"
            " - Keep the leading speaker tag exactly as [S1] or [S2].\n"
            " - If the line contains the placeholder [insert-verbal-tag-for-pause], "
            "replace it with one appropriate verbal tag (choose from the provided set) "
            "and do not add any others.\n"
            " - You may sparsely (<=15% of lines) add a short verbal tag to the start "
            "of the spoken text (immediately after the speaker tag) when context "
            "suggests it (e.g., (gasps), (laughs), …um,).\n"
            " - Do NOT overuse tags; maintain naturalness and vary the chosen tag.\n"
            " - Do not alter the main semantic content other than inserting/replacing "
            "verbal tags.\n"
            " - Output must be a single transcript line starting with the speaker tag."
        )

        default_human_prompt_template = (
            "Prev lines:\n{prev_lines}\n\n"
            "Current line:\n{current_line}\n\n"
            "Next lines:\n{next_lines}\n\n"
            "Conversation summary (short): {summary}\n"
            "Current topic: {topic}\n\n"
            "Available verbal tags (example set): {verbal_tags}\n\n"
            "Return only the modified single line."
        )

        self.VERBAL_TAG_INJECTOR_SYSTEM_PROMPT = prompts_config.get(
            "verbal_tag_injector", {}
        ).get("system_prompt", default_system_prompt)

        self.VERBAL_TAG_INJECTOR_HUMAN_PROMPT_TEMPLATE = prompts_config.get(
            "verbal_tag_injector", {}
        ).get("human_prompt_template", default_human_prompt_template)


# Global configuration instance
config = Config()
