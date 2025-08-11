# Bom Dia - Podcast Generator

Bom Dia is a tool that converts text transcripts into podcast-style audio files with verbal tags (like laughter, sighs, etc.) added for a more natural listening experience.

## Features

- Converts text transcripts to audio podcasts
- Automatically adds verbal tags for a more natural conversation flow
- Uses AI models for intelligent tag placement
- Employs an agentic workflow with Director and Actor agents for sophisticated tag injection
- Provider-agnostic LLM support through LiteLLM (OpenAI, Ollama, Anthropic, Google Gemini, etc.)

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd bomdia

# Install dependencies
pip install -e .
```

## Usage

```bash
bomdia input_transcript.txt output_podcast.mp3
```

### Verbosity Control

The application provides two flags for controlling output verbosity:

- `-v`, `--verbose`: Sets logging level to INFO, showing standard process flow
- `--verbosity {DEBUG,INFO,WARNING,ERROR}`: Sets a specific logging level

Examples:
```bash
# Show standard process flow
bomdia -v input_transcript.txt output_podcast.mp3

# Show detailed debugging information
bomdia --verbosity DEBUG input_transcript.txt output_podcast.mp3

# Show only errors and warnings (default)
bomdia input_transcript.txt output_podcast.mp3
```

## Configuration

### LLM Configuration

This project uses LiteLLM to support various LLM providers. Configure your model in `config/app.toml`:

```toml
[model]
# LiteLLM model string.
# OpenAI example: "openai/gpt-4o-mini"
# Ollama example: "ollama/llama3"
llm_spec = "ollama/llama3"

[model.parameters]
# Optional parameters to pass to the model
temperature = 0.5
max_tokens = 150
```

**Environment Variables:**
- For **OpenAI**, set `OPENAI_API_KEY`.
- For **Ollama**, ensure your Ollama server is running. If it's not at the default `http://localhost:11434`, set `OLLAMA_API_BASE`.
- For other providers (Anthropic, Gemini, etc.), see the LiteLLM documentation for required environment variables.

### Other Configuration

Other configuration options can be found in `config/app.toml`, including:
- Model checkpoints for the TTS system
- Pipeline behavior parameters
- Verbal tags and line combiners

Detailed information about the agentic workflow configuration can be found in `config/prompts.toml`.

## Development

### Running Tests

```bash
pytest
```

### Code Quality

```bash
# Format code
black .

# Lint code
ruff check .

# Type check
mypy src shared
```

## Components

For detailed information about specific components, see the [components documentation](docs/components/):

- [LiteLLM Integration](docs/components/litellm_integration.md) - Provider-agnostic LLM backend
- [Audio Generator](src/components/audio_generator/) - Text-to-speech conversion
- [Transcript Parser](src/components/transcript_parser/) - Transcript parsing and processing
- [Verbal Tag Injector](docs/components/verbal_tag_injector.md) - Agentic verbal tag injection logic

## External Dependency Warnings

See [EXTERNAL_DEPENDENCY_WARNINGS.md](EXTERNAL_DEPENDENCY_WARNINGS.md) for information about expected warnings from external dependencies.
