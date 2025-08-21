# Transcript Parser

## Overview

The Transcript Parser component is responsible for ingesting and parsing various transcript formats into a standardized internal representation. It supports both simple text files with speaker annotations and SRT subtitle files.

## Purpose and Scope

The Transcript Parser component enables the pipeline to work with different input transcript formats by converting them into a consistent structure that can be processed by subsequent components.

Key features:
- Support for multiple transcript formats (simple text, SRT)
- Automatic speaker identification and tagging
- Text normalization and cleaning
- Handling of consecutive lines from the same speaker

## Key Interfaces and APIs

### Core Functions

#### `ingest_transcript(path)`
Main entry point for parsing transcripts. Automatically detects the file format based on extension and calls the appropriate parser.

**Parameters:**
- `path` (str): Path to the transcript file

**Returns:** List of dictionaries with 'speaker' and 'text' keys

#### `parse_simple_txt(path)`
Parses simple text files with speaker annotations, intelligently extracting speaker IDs, optional speaker names, and dialogue text. Fully backward-compatible with legacy formats while supporting enhanced speaker identification.

Supports multiple line formats:
1. `[S1: John] Hello there.` - Named speaker tag (preferred format for maximum context)
2. `[S2] I'm doing well.` - Simple speaker tag (original format, fully supported)
3. `John: How are you?` - Colon-delimited name (parser assigns automatic speaker ID)
4. `bare text` - Continuation line (treated as part of previous speaker's dialogue)

**Parameters:**
- `path` (str): Path to the text file to parse

**Returns:** List of dictionaries with 'speaker' and 'text' keys

#### `parse_srt(path)`
Parses SRT subtitle files, extracting text content while ignoring timestamps. All lines are attributed to a default speaker.

**Parameters:**
- `path` (str): Path to the .srt file to parse

**Returns:** List of dictionaries with 'speaker' (default 'S1') and 'text' keys

#### `merge_consecutive_lines(lines)`
Merges consecutive lines from the same speaker with a pause placeholder.

**Parameters:**
- `lines` (List[Dict]): List of dictionaries with 'speaker' and 'text' keys

**Returns:** List of dictionaries with merged lines where appropriate

## Dependencies and Integration Points

- **Shared Config**: Uses `PAUSE_PLACEHOLDER` configuration for merging consecutive lines
- **Pipeline**: Integrated as the first step in the processing pipeline
- **Verbal Tag Injector**: Receives parsed and merged transcript lines for processing

## Configuration Requirements

The component requires no special configuration beyond what is provided by the shared configuration system.

## Usage Examples

### Simple Text Format

The parser supports multiple formats for flexible transcript creation:

```
[S1: Alice] Hello there, how are you today?
John: I'm doing well, thanks for asking!
[S2] That's great to hear.
I was wondering if you'd like to join us for lunch?
Yeah, that sounds like a plan.
```

**Format Breakdown:**
- `[S1: Alice] Hello there, how are you today?` - Named speaker tag (preferred format)
- `John: I'm doing well, thanks for asking!` - Colon-delimited name (automatic speaker ID assigned)
- `[S2] That's great to hear.` - Simple speaker tag (original format)
- `I was wondering if you'd like to join us for lunch?` - Continuation line (part of S2's dialogue)
- `Yeah, that sounds like a plan.` - Another continuation line (still part of S2's dialogue)

### SRT Format
```
1
00:00:01,000 --> 00:00:05,000
Hello there

2
00:00:05,500 --> 00:00:10,000
How are you?
```

## Error Handling

The component handles common file I/O errors and format-specific parsing issues gracefully:
- File not found errors
- Encoding issues with UTF-8 fallback
- Malformed SRT files
- Invalid speaker annotations

## Testing

The component includes comprehensive tests for:
- Different transcript formats
- Edge cases in parsing
- Speaker identification logic
- Line merging functionality
