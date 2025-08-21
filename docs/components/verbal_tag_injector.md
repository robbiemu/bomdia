# Verbal Tag Injector Component

## Overview

The Verbal Tag Injector component is responsible for intelligently injecting verbal tags into a transcript. It uses a sophisticated Moment-Based Acting Architecture to enhance the Director/Actor workflow with narrative intelligence and global pacing controls.

This component can be bypassed entirely by using the `--no-rehearsals` command-line flag, which skips the agentic workflow and processes clean, pre-formatted transcripts directly.

## Architecture

The core of this component is the Director/Actor model.

### Director
The Director's behavior is guided by a consistent `system_prompt` to establish its role.

The Director is responsible for:
-   **Narrative Moment Discovery**: Identifying narrative moments based on consistent topic, intention, and emotional tone.
-   **Pivot Line Handling**: Allowing single lines to be part of multiple moments.
-   **Global Pacing**: Managing a central token bucket to control the rate of tag injection across the entire script.
-   **Orchestration**: Driving the rehearsal process, defining moments, and delegating to the Actor.

### Actor
The Actor's behavior is also guided by a `system_prompt` to ensure it performs its role as a voice actor correctly.

The Actor is responsible for:
-   **Creative Performance**: Generating the actual verbal tags for a given moment.
-   **Moment-at-once Processing**: Processing an entire moment in a single, efficient LLM call.

### Workflow

1.  **Setup**: The Director initializes the transcript, generates a global summary, and sets up the token bucket for pacing. All LLM calls are guided by system prompts.
2.  **Graph-Based Rehearsal**: The Director's orchestration logic is managed by a **LangGraph state machine**. This graph iterates through the script's state (`current_line_index`), discovering and defining moments in a persistent and resumable workflow.
3.  **Performance**: When a moment is complete, the Director delegates to the Actor to perform the moment.
4.  **Director's Final Cut**: The Director reviews the Actor's performance through one of two modes:
    -   **Procedural Mode** (default): Fast, rule-based tag pruning that removes the last N newly added tags to meet the budget.
    -   **LLM Mode**: Slower, higher-quality LLM-based review that uses contextual intelligence to choose which tags to keep or remove.
5.  **Fallback Protection**: If LLM mode fails, the system automatically falls back to procedural mode with detailed logging.
6.  **Recomposition**: The reviewed lines are placed back into the final script.
7.  **Final Cleanup**: Before finishing, a final pass removes any leftover technical placeholders to ensure a clean output script.

## Director's Final Cut

The Director's Final Cut is a quality control phase that ensures Actor performances comply with strict budget constraints while preserving the most impactful verbal tags.

### Review Modes

#### Procedural Mode

-   **Speed**: Fast, algorithmic processing
-   **Logic**: Removes the last N newly added tags in reading order to meet budget
-   **Use Case**: Environments where speed is critical or as a reliable fallback.
-   **Reliability**: Deterministic, always produces budget-compliant results

#### LLM Mode

-   **Speed**: Slower due to additional LLM call
-   **Logic**: Uses contextual intelligence to choose which tags to preserve based on:
    -   Narrative importance
    -   Character motivation
    -   Emotional impact
    -   Global script context
-   **Use Case**: High-quality productions where nuanced tag selection is valuable
-   **Reliability**: Includes automatic fallback to procedural mode on failure

### Configuration

```toml
[director_agent.review]
mode = "llm"  # or "procedural"
```

Environment override: `DIRECTOR_AGENT_REVIEW_MODE=llm`

### Routing and Fallback

1. **Mode Selection**: Based on configuration, routes to appropriate review node
2. **LLM Review**: If selected, attempts intelligent tag curation
3. **Fallback Protection**: On LLM failure (network issues, malformed JSON, etc.), automatically falls back to procedural mode
4. **Error Logging**: All fallback events are logged with detailed error information for debugging

## Key Features

-   **Deterministic Tag Budgeting**: The number of new tags is calculated based on the script length.
-   **Quality Control**: Director's Final Cut ensures budget compliance while maximizing tag value.
-   **Robust Error Recovery**: The system can handle errors from the LLM, such as invalid JSON or moment boundaries.
-   **Performance Optimization**: The number of LLM calls is minimized by processing entire moments at once.
-   **Narrative Intelligence**: Moments are defined based on the story's logic, not just technical boundaries.
-   **Prompt Configuration**: All prompts are externally configurable through `config/prompts.toml`, allowing customization of the AI behavior without code changes.
