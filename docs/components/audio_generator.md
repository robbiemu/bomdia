# Audio Generator

## Overview

The Audio Generator component is responsible for converting processed transcript text into audio using the Dia TTS (Text-to-Speech) model. It handles voice cloning, audio generation, and chunking of longer transcripts into appropriate segments.

## Purpose and Scope

The Audio Generator component transforms the processed textual podcast into actual audio files. It manages the complex process of generating high-quality speech from text while supporting voice cloning capabilities.

Key features:
- Text-to-speech conversion using the Dia model
- Voice cloning from audio prompts
- Transcript chunking for optimal audio generation
- Multi-speaker support
- Cross-platform device support (CUDA, MPS, CPU)

## Key Interfaces and APIs

### Core Classes

#### `DiaTTS`
Main class for text-to-speech conversion using the Dia library.

##### `__init__(seed, model_checkpoint, device, log_level)`
Initializes the TTS engine using the Dia library.

**Parameters:**
- `seed` (int, optional): Random seed for reproducibility
- `model_checkpoint` (str): The Hugging Face model identifier
- `device` (str, optional): Device to run on ('cuda', 'mps', or 'cpu')
- `log_level` (str, optional): Logging level for the TTS model's verbose output ('DEBUG', 'INFO', 'WARNING', 'ERROR')

##### `register_voice_prompts(voice_prompts)`
Analyzes audio files to generate and store speaker embeddings.

**Parameters:**
- `voice_prompts` (Dict[str, str]): Dictionary mapping speaker tags to audio file paths

##### `text_to_audio_file(text, out_path)`
Converts text to an audio file using the Dia model.

**Parameters:**
- `text` (str): Text to convert to speech
- `out_path` (str): Path to save the output audio file

### Core Functions

#### `chunk_to_5_10s(lines)`
Combines line strings into blocks of 5-10 seconds with edge case handling.

Each block is a newline-separated transcript with speaker tags preserved.

**Parameters:**
- `lines` (List[dict]): List of dictionaries with 'speaker' and 'text' keys

**Returns:** List of transcript blocks as strings, each 5-10 seconds long

#### `estimate_seconds_for_text(text)`
Estimates the time in seconds for a given text based on average words per second.

**Parameters:**
- `text` (str): Text to estimate timing for

**Returns:** Estimated time in seconds

## Dependencies and Integration Points

- **Dia Library**: Core dependency for TTS functionality
- **PyTorch**: Deep learning framework for model execution
- **Shared Config**: Loads timing and chunking parameters
- **Pipeline**: Integrated as the final step in the processing pipeline
- **PyDub**: Audio file concatenation

## Configuration Requirements

### Configuration Values

The component uses several configuration values from `config/app.toml`:

- `DIA_CHECKPOINT`: Model checkpoint for the Dia TTS model
- `DIA_COMPUTE_DTYPE`: Compute data type for the model (float16, float32, bfloat16)
- `AVG_WPS`: Average words per second for timing estimation
- `SEED`: Random seed for reproducible voice selection
- `DIA_GENERATE_PARAMS`: Dictionary of parameters for the Dia TTS generation (only parameters explicitly set will be passed to the model)

## Usage Examples

### Basic Usage
```python
tts = DiaTTS(seed=42)
audio_path = tts.text_to_audio_file("Hello, world!", "output.wav")
```

### Voice Cloning
```python
tts = DiaTTS(seed=42)
tts.register_voice_prompts({
    "S1": "speaker1.wav",
    "S2": "speaker2.wav"
})
audio_path = tts.text_to_audio_file("[S1] Hello there! [S2] Hi!", "output.wav")
```

### Chunking
```python
lines = [
    {"speaker": "S1", "text": "Hello there"},
    {"speaker": "S2", "text": "How are you?"},
]
blocks = chunk_to_5_10s(lines)
# Process each block separately
```

## Error Handling

The component handles various error conditions:
- Model loading failures
- Audio generation errors
- Voice cloning issues
- File I/O problems
- Device compatibility issues

Specific troubleshooting guidance is provided for common issues like:
- CUDA/MPS device configuration
- Memory constraints
- Audio file format problems

## Testing

The component includes comprehensive tests:
- Unit tests for chunking algorithms
- Timing estimation accuracy
- Voice cloning functionality
- Error condition handling
- Integration tests with the pipeline

Note that TTS generation tests are mocked to avoid requiring the full model during testing.
