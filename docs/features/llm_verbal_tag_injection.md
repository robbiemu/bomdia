# LLM-Powered Verbal Tag Injection

## Overview

This feature enables intelligent verbal tag injection by leveraging large language models through the LiteLLM library. It processes transcript lines with contextual awareness to add appropriate verbal tags (like laughter, sighs, etc.) for a more natural podcast experience.

## Purpose and Scope

The LLM-powered verbal tag injection feature enhances the naturalness of generated podcasts by using large language models to determine optimal placement of verbal tags. Rather than relying on simple rule-based approaches, this feature uses contextual understanding to make more sophisticated decisions about where and how to insert verbal tags.

Key capabilities:
- Context-aware verbal tag placement
- Support for multiple LLM providers
- Configurable model parameters
- Graceful fallback behavior for error handling
- Extensibility to new providers without code changes

## Configuration

### Model Specification

The feature is configured through `config/app.toml`:

```toml
[model]
# LiteLLM model string (e.g., "openai/gpt-4o-mini", "ollama/llama3")
llm_spec = "openai/gpt-4o-mini"

[model.parameters]
# Optional parameters to pass to the model
temperature = 0.5
max_tokens = 150
```

The `llm_spec` follows LiteLLM's model string format:
- OpenAI: `openai/gpt-4o-mini`
- Ollama: `ollama/llama3`
- Anthropic: `anthropic/claude-3-haiku-20240307`
- Google: `gemini/gemini-1.5-pro-latest`

### Environment Variables

- `LLM_SPEC`: Overrides the model specification in config
- Provider-specific API keys (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`)
- `OLLAMA_API_BASE`: Custom Ollama endpoint (default: http://localhost:11434)

### Model Parameters

Optional parameters can be specified in `[model.parameters]`:
- `temperature`: Controls randomness (0.0-1.0)
- `max_tokens`: Maximum tokens to generate
- Any other provider-specific parameters

## Usage Examples

### Cloud Provider (OpenAI)

```toml
[model]
llm_spec = "openai/gpt-4o-mini"

[model.parameters]
temperature = 0.5
max_tokens = 150
```

Set `OPENAI_API_KEY` environment variable.

### Local Model (Ollama)

```toml
[model]
llm_spec = "ollama/llama3"

[model.parameters]
temperature = 0.7
```

Ensure Ollama is running locally.

### Adding New Providers

To add support for a new provider:
1. Set the required environment variables (API keys)
2. Update `llm_spec` to the appropriate model string
3. Add any required parameters to `[model.parameters]`

No code changes are required for new providers supported by LiteLLM.

## Error Handling

The feature gracefully handles API errors by falling back to rule-based tag injection. Errors are logged to the console but don't interrupt the pipeline processing.

## Testing

The feature includes comprehensive tests:
- Mock-based unit tests for API interaction
- Error handling verification
- Parameter passing validation
- Integration tests with the pipeline

All tests use mocking to avoid real API calls.
