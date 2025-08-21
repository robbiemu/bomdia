import logging
from dataclasses import dataclass
from typing import Any, Dict, List

import litellm

# Try to import the traceable decorator. If langsmith is not installed,
# create a dummy decorator that does nothing.
try:
    from langsmith import traceable

    from .litellm_callbacks import setup_litellm_callbacks

    # set up LangSmith tracing with litellm
    setup_litellm_callbacks()
except ImportError:
    # This function is a placeholder that mimics the @traceable decorator.
    # It takes the same arguments but returns a simple wrapper that just
    # returns the original function, effectively doing nothing.
    def dummy_traceable(*_args: Any, **_kwargs: Any) -> Any:
        def wrapper(func: Any) -> Any:
            return func

        return wrapper

    traceable = dummy_traceable


# Initialize logger
logger = logging.getLogger(__name__)


# Step 1: Define a simple, predictable response structure.
# This mimics LangChain's AIMessage, giving us the clean `response.content` access.
@dataclass
class LLMResponse:
    """A standardized, simple response object for LLM calls."""

    content: str


# Step 2: Create the Invoker class.
class LiteLLMInvoker:
    """
    A wrapper for LiteLLM that provides a simple, LangChain-like .invoke() method.
    """

    def __init__(self, model: str, **kwargs: Any):
        """
        Initializes the invoker with a specific model and optional default parameters.

        Args:
            model: The name of the model to use (e.g., "gpt-4o-mini").
            **kwargs: Default parameters for litellm.completion (e.g., temperature,
                max_tokens).
        """
        self.model = model
        self.default_params = kwargs

    @traceable(run_type="llm")
    def invoke(self, messages: List[Dict[str, str]]) -> LLMResponse:
        """
        Invokes the LLM with a list of messages and returns a standardized response.

        Args:
            messages: A list of message dictionaries, e.g., [
                {"role": "user", "content": "..."}
            ].

        Returns:
            An LLMResponse object containing the model's response content.
        """
        try:
            # Combine default params with the specific call params
            params = {"model": self.model, "messages": messages, **self.default_params}

            response = litellm.completion(**params)

            # Safely parse the response content
            content = response.choices[0].message.content or ""

            return LLMResponse(content=content.strip())

        except Exception as e:
            logger.error(f"LiteLLM call to model '{self.model}' failed: {e}")
            # Return a standardized empty response on failure
            return LLMResponse(content="")
