from dataclasses import dataclass
from typing import Any, Dict, List

import litellm


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
            print(f"ERROR: LiteLLM call to model '{self.model}' failed: {e}")
            # Return a standardized empty response on failure
            return LLMResponse(content="")
