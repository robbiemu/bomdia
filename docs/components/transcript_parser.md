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
Parses simple text files with speaker annotations.

Supports three line formats:
1. `[S1] text...` - Explicit speaker tags in brackets
2. `Name: text...` - Name-based speaker identification
3. `bare text` - Assumed to be continuation of previous speaker or S1 if first line

**Parameters:**
- `path` (str): Path to the text file to parse

**Returns:** List of dictionaries with 'speaker' and 'text' keys

#### `parse_srt(path)`
Parses SRT subtitle files, extracting text content while ignoring timestamps.

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
```
[S1] Hello there
John: How are you?
[S2] I'm doing well
Nice to hear
```

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
