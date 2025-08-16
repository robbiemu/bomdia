# Observability Feature

The observability feature provides comprehensive insight into the application's behavior through two distinct layers: a unified logging system and optional LangSmith tracing.

## Feature Overview

Observability is crucial for understanding how the application processes transcripts and generates podcasts. This feature enables both end-users and developers to monitor the application's behavior at different levels of detail.

## Unified Logging System

The core of the observability feature is a unified logging system that replaces all previous `print()` statements with properly structured log messages.

### Benefits

- **Configurable Verbosity**: Users can control the amount of output through CLI flags
- **Structured Output**: All messages follow a consistent format with timestamps and log levels
- **Enhanced Transparency**: Rich INFO-level logging provides insights into the creative process
- **Standard Interface**: Uses Python's standard `logging` module for familiarity and compatibility

### Log Levels

The application uses four standard log levels:

- **ERROR**: Critical issues that prevent normal operation
- **WARNING**: Non-critical issues that users should be aware of
- **INFO**: Standard process flow information for normal operation, including detailed insights into the creative process
- **DEBUG**: Detailed information for troubleshooting and development

At the INFO level, users can now see:
- The script being sent to the Actor agent
- The Actor's creative "take" on each moment
- The Director's final reviewed script after quality control

### CLI Controls

Users can control verbosity through two CLI flags:

```bash
# Basic verbosity control
bomdia -v input.txt output.mp3          # INFO level
bomdia --verbosity DEBUG input.txt output.mp3  # DEBUG level

# Default is WARNING level (minimal output)
bomdia input.txt output.mp3
```

## LangSmith Tracing

For developers and advanced users, the application includes optional LangSmith tracing integration.

### Benefits

- **Detailed LLM Tracing**: Track all LLM calls and their inputs/outputs
- **Agent Workflow Visualization**: See the Director-Actor workflow execution
- **Performance Monitoring**: Monitor execution time and resource usage
- **No Configuration Required**: Enabled purely through environment variables

### Activation

LangSmith tracing is activated by setting environment variables:

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
export LANGCHAIN_API_KEY=your-langsmith-api-key
export LANGCHAIN_PROJECT=your-project-name
```

When these variables are not set, the application runs normally without any tracing overhead.

## Architecture

The observability system is designed with a clear separation of concerns:

1. **Application Logging**: Built-in feature for all users
2. **Developer Tracing**: Optional feature for detailed analysis

This separation ensures that:
- End users get the information they need without complexity
- Developers can access detailed tracing when needed
- The application performs well regardless of tracing configuration

## Implementation Details

### Logger Configuration

The logging system is configured at application startup in `main.py`:

```python
# Determine logging level from CLI flags
log_level = getattr(logging, args.verbosity) if args.verbosity != "WARNING" else logging.WARNING
log_level = logging.INFO if args.verbose and args.verbosity == "WARNING" else log_level

# Initialize logger
setup_logger(log_level)
```

### Component Integration

All components use the standard logging approach:

```python
import logging
logger = logging.getLogger(__name__)

# In code:
logger.info("Processing transcript block %d/%d", i+1, len(blocks))
logger.debug("LLM response: %s", response.content)
logger.warning("Falling back to default voice due to missing prompt")
logger.error("Failed to generate audio: %s", str(e))
```

This approach ensures consistent logging behavior across all parts of the application.
