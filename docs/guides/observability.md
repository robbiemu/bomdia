# Observability Guide

This guide outlines the logging strategy for the application, designed to provide clear insights into the agent's decision-making process.

## Logging Levels

The logging system uses standard logging levels to provide different levels of detail:

-   **INFO**: High-level, user-facing messages that tell the story of the creative process.
-   **DEBUG**: Detailed information for developers, including internal decisions, constraints, and budget status.
-   **WARNING**: Notifications about error recovery events, such as invalid data from the LLM.

## Key Log Messages

### INFO Level

-   `Starting rehearsal graph execution (thread_id: ...)`
-   `--- Processing Moment {moment_id} ---`
-   `Moment description: {description}`
-   `Moment director's notes: {directors_notes}`
-   `Moment {moment_id} finalized in {duration:.2f}s.`
-   `--- Rehearsal graph completed successfully...`
-   `--- Moment-based rehearsal process complete. ---`

### DEBUG Level

-   `Moment {moment_id}: Token budget for Actor is {budget:.1f} tokens.`
-   `Global tag budget status: {used}/{total}.`
-   `Moment {moment_id}: Skipping due to exhausted global tag budget...`
-   `Forward-cascading constraint applied to pivot line...`

### WARNING Level

-   `Invalid moment boundaries for moment {moment_id}. Creating fallback single-line moment.`
-   `Recovery from JSON parsing error in moment definition. LLM response was: '{error_details...}'`
