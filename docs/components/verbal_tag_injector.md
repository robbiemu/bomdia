# Verbal Tag Injector Component

## Overview

The Verbal Tag Injector component is responsible for intelligently injecting verbal tags into a transcript. It uses a sophisticated Moment-Based Acting Architecture to enhance the Director/Actor workflow with narrative intelligence and global pacing controls.

## Architecture

The core of this component is the Director/Actor model.

### Director

The Director is responsible for:
-   **Narrative Moment Discovery**: Identifying narrative moments based on consistent topic, intention, and emotional tone.
-   **Pivot Line Handling**: Allowing single lines to be part of multiple moments.
-   **Global Pacing**: Managing a central token bucket to control the rate of tag injection across the entire script.
-   **Orchestration**: Driving the rehearsal process, defining moments, and delegating to the Actor.

### Actor

The Actor is responsible for:
-   **Creative Performance**: Generating the actual verbal tags for a given moment.
-   **Moment-at-once Processing**: Processing an entire moment in a single, efficient LLM call.

### Workflow

1.  **Setup**: The Director initializes the transcript, generates a global summary, and sets up the token bucket for pacing.
2.  **Graph-Based Rehearsal**: The Director's orchestration logic is managed by a **LangGraph state machine**. This graph iterates through the script's state (`current_line_index`), discovering and defining moments in a persistent and resumable workflow.
3.  **Performance**: When a moment is complete, the Director delegates to the Actor to perform the moment.
4.  **Review**: The Director reviews the Actor's performance and finalizes the moment.
5.  **Recomposition**: The edited lines are placed back into the final script.

## Key Features

-   **Deterministic Tag Budgeting**: The number of new tags is calculated based on the script length.
-   **Robust Error Recovery**: The system can handle errors from the LLM, such as invalid JSON or moment boundaries.
-   **Performance Optimization**: The number of LLM calls is minimized by processing entire moments at once.
-   **Narrative Intelligence**: Moments are defined based on the story's logic, not just technical boundaries.
-   **Prompt Configuration**: All prompts are externally configurable through `config/prompts.toml`, allowing customization of the AI behavior without code changes.
