"""
Centralized setup for LiteLLM callbacks, ensuring LangSmith tracing is active.
"""

import asyncio
import os
from typing import Any, Optional

import litellm
from litellm import CustomLogger
from litellm.integrations.langsmith import LangsmithLogger

from shared.logging import get_logger

logger = get_logger(__name__)

_litellm_logger_instance: Optional["DeferredLangsmithLogger"] = None


class DeferredLangsmithLogger(CustomLogger):
    """
    Delay scheduling the background task until an event loop is actually running.
    If none is found, batching will be handled synchronously.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._args = args
        self._kwargs = kwargs
        self._logger: Optional[LangsmithLogger] = None
        self._task_started = False

    def _ensure_logger(self) -> None:
        if not self._logger:
            self._logger = LangsmithLogger(*self._args, **self._kwargs)

    def _maybe_schedule(self) -> None:
        if not self._task_started:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return
            if self._logger:
                loop.create_task(self._logger.periodic_flush())
                self._task_started = True

    def log_event(self, *args: Any, **kwargs: Any) -> Any:
        self._ensure_logger()
        self._maybe_schedule()
        if self._logger:
            return self._logger.log_event(*args, **kwargs)
        return None

    def log_success_event(self, *args: Any, **kwargs: Any) -> Any:
        self._ensure_logger()
        self._maybe_schedule()
        if self._logger:
            return self._logger.log_success_event(*args, **kwargs)
        return None

    def log_failure_event(self, *args: Any, **kwargs: Any) -> Any:
        self._ensure_logger()
        self._maybe_schedule()
        if self._logger:
            return self._logger.log_failure_event(*args, **kwargs)
        return None


def setup_litellm_callbacks() -> None:
    """Initializes LiteLLM callbacks based on environment variables."""
    global _litellm_logger_instance

    tracing_enabled = (
        os.getenv("LANGCHAIN_TRACING_V2") == "true"
        or os.getenv("LANGSMITH_TRACING") == "true"
    )
    api_key_present = os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY")

    if tracing_enabled and api_key_present:
        logger.debug(
            "LangSmith tracing is enabled. Setting 'langsmith' as a LiteLLM callback."
        )
        _litellm_logger_instance = DeferredLangsmithLogger()
        litellm.callbacks = [_litellm_logger_instance]
    else:
        logger.debug("LangSmith tracing not enabled. Skipping LiteLLM callback setup.")
