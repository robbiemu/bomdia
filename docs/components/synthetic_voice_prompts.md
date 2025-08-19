# Synthetic Voice Prompt Generation

## Overview

The synthetic voice prompt generation feature automatically creates voice prompts for speakers that don't have user-provided audio samples. This improves voice consistency and quality in the final podcast output by ensuring all speakers have voice references during the TTS generation process.

## Architecture

### Components

1. **Worker Script (`generate_prompt.py`)**: Standalone script for generating individual synthetic voice prompts
2. **Pipeline Integration**: Automatic detection and generation of missing voice prompts during the main pipeline execution
3. **Configuration System**: Settings to control synthetic prompt generation behavior

### Workflow

```
Input Transcript → Parse Speakers → Detect Missing Prompts → Generate Synthetic Prompts → Continue Pipeline
```

## Worker Script (`generate_prompt.py`)

### Purpose

The worker script is a standalone executable that generates a single synthetic voice prompt for a specified speaker. It operates independently of the main pipeline and can be used for:

- Pre-generating voice prompts
- Testing voice generation
- Manual voice prompt creation

### Arguments

- `--speaker-id`: Speaker identifier (e.g., "S1", "S2")
- `--seed`: Random seed for reproducible generation (optional)
- `--output-dir`: Directory to save generated files (default: current directory)
- `--verbose`: Enable verbose logging

### Output

The script generates:

1. **Audio File**: `{speaker_id}_{seed}.wav` - Synthetic voice prompt audio
2. **Transcript File**: `{speaker_id}_{seed}.txt` - Corresponding transcript text
3. **JSON Metadata**: Printed to stdout with speaker_id, audio_path, and stdout_transcript

### Example Usage

```bash
# Generate prompt for speaker S1 with specific seed
generate-prompt --speaker-id S1 --seed 12345 --output-dir ./voice_prompts/

# Generate with random seed
generate-prompt --speaker-id S2 --output-dir ./voice_prompts/ --verbose
```

## Pipeline Integration

### Detection Phase

The pipeline automatically detects speakers that lack voice prompts by:

1. Parsing the input transcript to identify all speakers
2. Checking which speakers have user-provided `--sX-voice` arguments
3. Creating a list of unprompted speakers

### Generation Phase

For each unprompted speaker:

1. **Subprocess Invocation**: Calls the worker script as a subprocess
2. **Seed Management**: Passes the pipeline seed or generates a unique seed
3. **File Collection**: Collects generated audio and transcript files
4. **Integration**: Adds synthetic prompts to the voice_prompts dictionary

### Error Handling

- **Worker Script Failures**: Logged as warnings, pipeline continues without synthetic prompts
- **File I/O Issues**: Handled gracefully with appropriate error messages
- **Missing Assets**: Fallback to pure TTS mode if generation text asset is missing

## Configuration

### app.toml Settings

```toml
[pipeline]
# Enable/disable synthetic voice prompt generation
generate_synthetic_voice_prompts = true  # default: true
```

### Generation Text Asset

The synthetic prompts use predefined text from:
```
assets/minimum_generation.one_speaker.txt
```

This file contains neutral text suitable for voice characteristic learning.

## Technical Details

### Audio Specifications

- **Format**: WAV (uncompressed)
- **Sample Rate**: 22,050 Hz (DiaTTS default)
- **Channels**: Mono
- **Duration**: Approximately 3-5 seconds

### File Naming Convention

Generated files follow the pattern:
- Audio: `{speaker_id}_{seed}.wav`
- Transcript: `{speaker_id}_{seed}.txt`

Where `seed` is an 8-digit zero-padded number.

### Transcript Processing

The worker script processes the generation text through three forms:

1. **File Transcript**: Raw text with speaker tag
2. **Stdout Transcript**: Newlines replaced with double spaces
3. **Generation Transcript**: Stdout transcript with continuity tag appended

## Testing

### Unit Tests

Located in `tests/test_generate_prompt.py`:

- Argument parsing validation
- File generation verification
- JSON output format checking
- Error handling scenarios
- Main function integration

### Integration Tests

Located in `tests/test_pipeline_synthetic_prompts.py`:

- Pipeline integration testing
- Subprocess mocking
- Error condition handling
- Configuration option testing

## Best Practices

### When to Use

- Multi-speaker podcasts without complete voice prompt sets
- Ensuring consistent voice quality across all speakers
- Improving voice cloning performance

### When to Disable

- Single-speaker content
- All speakers have high-quality voice prompts
- Performance-critical environments where generation time matters

### Performance Considerations

- Synthetic prompt generation adds processing time
- Each prompt generation requires DiaTTS model loading
- Consider pre-generating prompts for frequently used speakers

## Troubleshooting

### Common Issues

1. **Worker Script Not Found**
   - Ensure `pip install -e .` was run to install the script
   - Check that `generate-prompt` is in your PATH

2. **Generation Failures**
   - Verify DiaTTS installation and model files
   - Check generation text asset file exists
   - Ensure sufficient disk space for output files

3. **Audio Quality Issues**
   - Review generation text content
   - Consider adjusting DiaTTS model parameters
   - Verify input text is appropriate for voice learning

### Debugging

Enable verbose logging with `--verbose` flag or set logging level to DEBUG to see detailed execution information.

## Future Enhancements

Potential improvements for the synthetic voice prompt system:

- **Adaptive Text Selection**: Choose generation text based on speaker characteristics
- **Quality Assessment**: Automatic evaluation of generated prompt quality
- **Caching System**: Reuse previously generated prompts for known speaker/seed combinations
- **Batch Generation**: Generate multiple speaker prompts in parallel
- **Custom Templates**: User-defined generation text for specific use cases
