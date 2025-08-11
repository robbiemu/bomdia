# Developer Guide: Using Observability Features

This guide explains how developers can leverage the observability features when implementing new functionality in the bomdia application.

## Understanding the Two Observability Layers

The bomdia application provides two distinct observability layers that serve different purposes:

1. **Application Logging**: Built-in structured logging for all application behavior
2. **LangSmith Tracing**: Optional detailed tracing for LLM interactions and agent workflows

## Using Application Logging

All new components should use the standard Python logging module rather than print statements.

### Basic Setup

In any new Python file, import and initialize the logger:

```python
import logging
logger = logging.getLogger(__name__)
```

### Choosing Appropriate Log Levels

Use the following guidelines when selecting log levels:

- **DEBUG**: Detailed information for troubleshooting, such as:
  - Raw LLM responses during development
  - Internal state transitions
  - Detailed parameter values

- **INFO**: General operational messages for normal execution, such as:
  - Component initialization
  - Processing progress updates
  - Successful completion of major steps

- **WARNING**: Non-critical issues that don't stop execution, such as:
  - Fallback behavior activation
  - Suboptimal configuration detected
  - Recoverable errors

- **ERROR**: Critical issues that prevent normal operation, such as:
  - Failed file operations
  - Invalid input data
  - LLM call failures

### Example Implementation

```python
import logging
logger = logging.getLogger(__name__)

class TranscriptProcessor:
    def __init__(self, config):
        self.config = config
        logger.info("Initializing transcript processor with model %s", config.model_name)

    def process_transcript(self, transcript_path):
        logger.info("Processing transcript: %s", transcript_path)

        try:
            transcript = self.load_transcript(transcript_path)
            logger.debug("Loaded transcript with %d lines", len(transcript))
        except FileNotFoundError:
            logger.error("Transcript file not found: %s", transcript_path)
            raise
        except Exception as e:
            logger.error("Failed to load transcript: %s", str(e))
            raise

        processed = self.enhance_transcript(transcript)
        logger.info("Transcript processing completed successfully")
        return processed

    def enhance_transcript(self, transcript):
        logger.debug("Starting transcript enhancement with %d lines", len(transcript))

        enhanced = []
        for i, line in enumerate(transcript):
            logger.debug("Processing line %d: %.50s...", i, line.get('text', ''))

            # Enhancement logic here
            enhanced_line = self.apply_enhancements(line)

            # Log significant changes or decisions
            if self.line_was_modified(line, enhanced_line):
                logger.info("Enhanced line %d with verbal tags", i)

            enhanced.append(enhanced_line)

        logger.debug("Completed enhancement of %d lines", len(enhanced))
        return enhanced
```

## Leveraging LangSmith Tracing

LangSmith tracing is particularly valuable when developing components that involve LLM interactions or complex agent workflows.

### When to Use LangSmith Tracing

LangSmith tracing is most beneficial for:

1. **LLM Interaction Development**: Tracking prompts, responses, and performance
2. **Agent Workflow Debugging**: Visualizing multi-step reasoning processes
3. **Prompt Engineering**: Comparing different prompt versions
4. **Performance Analysis**: Identifying bottlenecks in AI-powered components

### Example: Movie Transcript Conversion Project

When implementing a system to convert movie transcripts to our format, you would leverage observability as follows:

```python
import logging
from langchain.callbacks import tracing_v2_enabled
from shared.llm_invoker import LiteLLMInvoker

logger = logging.getLogger(__name__)

class MovieTranscriptConverter:
    def __init__(self):
        self.llm = LiteLLMInvoker(model="openai/gpt-4o-mini")
        logger.info("Initialized movie transcript converter")

    def convert_transcript(self, movie_transcript_path):
        logger.info("Starting conversion of movie transcript: %s", movie_transcript_path)

        # Load and parse the movie transcript
        raw_transcript = self.load_movie_transcript(movie_transcript_path)
        logger.debug("Loaded %d lines from movie transcript", len(raw_transcript))

        # Process each line with LLM assistance
        converted_lines = []
        for i, line in enumerate(raw_transcript):
            logger.info("Converting line %d/%d", i+1, len(raw_transcript))

            # Use LangSmith tracing for the LLM interaction
            with tracing_v2_enabled():
                converted_line = self.convert_line_format(line)

            converted_lines.append(converted_line)

            # Periodic progress logging
            if (i + 1) % 100 == 0:
                logger.info("Converted %d/%d lines", i+1, len(raw_transcript))

        logger.info("Completed conversion of %d lines", len(converted_lines))
        return converted_lines

    def convert_line_format(self, line):
        logger.debug("Converting line format: %.100s", line.get('text', ''))

        prompt = self.build_conversion_prompt(line)
        logger.debug("Sending prompt to LLM (length: %d chars)", len(prompt))

        try:
            response = self.llm.invoke([{"role": "user", "content": prompt}])
            logger.debug("Received LLM response (length: %d chars)", len(response.content))

            converted = self.parse_response(response.content)
            logger.debug("Parsed response into structured format")

            return converted
        except Exception as e:
            logger.error("Failed to convert line: %s", str(e))
            # Return fallback format
            return self.create_fallback_format(line)
```

### Analyzing LangSmith Traces

With LangSmith tracing enabled, you can:

1. **Compare Prompt Versions**: Track how different prompts affect output quality
2. **Monitor Token Usage**: Understand the cost implications of your LLM calls
3. **Debug Failures**: Examine exact inputs that led to errors
4. **Optimize Performance**: Identify slow operations in your workflow

## Best Practices

### 1. Log Early and Often
Don't wait until something goes wrong to add logging. Add informative log messages during development to help future debugging.

### 2. Use Structured Logging
Include relevant context in your log messages:

```python
# Good - includes context
logger.info("Processed audio segment %d/%d (duration: %.2fs)",
            i+1, len(segments), segment.duration)

# Avoid - lacks context
logger.info("Processed segment")
```

### 3. Protect Sensitive Information
Never log API keys, passwords, or other sensitive data:

```python
# Bad - logs sensitive information
logger.debug("API key: %s", api_key)

# Good - logs only non-sensitive information
logger.debug("Using API key with prefix: %s...", api_key[:8])
```

### 4. Balance Verbosity
Use appropriate log levels to avoid overwhelming users while providing enough information for debugging:

```python
# At INFO level - general progress
logger.info("Starting audio generation for %d segments", len(segments))

# At DEBUG level - detailed information
logger.debug("Generating audio for segment: %s (speaker: %s, duration: %.2fs)",
             text[:50], speaker, estimated_duration)
```

### 5. Test Your Logging
Verify that your log messages appear as expected when running the application with different verbosity levels.

## Testing with Different Log Levels

During development, test your components with different verbosity settings:

```bash
# Default (WARNING) - minimal output
python -m src.your_module

# INFO level - standard operational messages
python -m src.your_module -v

# DEBUG level - detailed debugging information
python -m src.your_module --verbosity DEBUG
```

This ensures your logging provides the right amount of information at each level.
