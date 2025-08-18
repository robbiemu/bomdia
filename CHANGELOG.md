# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [0.1.7]

### Changed
- **Post-processes transcript chunks*** for continuity
- **Conversion to Batch mode processing** Making use of the batch mode processing provided by Dia's generate() method.

### Fixed
- **Dead clode cleanup** Some code and configuration that was unused has been removed entirely.

## [0.1.6] - 2025-08-16

### Added
- **Dry Run Mode**: A new `--dry-run` flag executes the entire agentic rehearsal process and prints the final, modified transcript to the console before exiting, skipping all audio generation steps.
- **No Rehearsals Mode**: A new `--no-rehearsals` flag provides a "fast path" that bypasses the entire Director/Actor workflow, sending the parsed transcript directly to the audio generation stage.
- **Configurable TTS Precision**: Added `dia_compute_dtype` setting in `config/app.toml` allowing users to specify the `torch.dtype` (e.g., "float16", "float32", "bfloat16") for the DiaTTS model.
- **Configurable TTS Generation Parameters**: Exposed key parameters of the `Dia.generate()` method in `config/app.toml` including `max_tokens`, `cfg_scale`, `temperature`, `top_p`, `cfg_filter_top_k`, and `use_cfg_filter`.
- **Configurable Device Selection**: Added `device` setting in `config/app.toml` allowing users to explicitly specify the compute device (`cuda`, `mps`, or `cpu`) for TTS model inference. Users can also override this with the `BOMDIA_DEVICE` environment variable. If neither is set, the application auto-detects the best available device in the priority order: `cuda` > `mps` > `cpu`.

### Changed
- **Enhanced INFO-Level Logging**: The rehearsal graph now logs the most critical inputs and outputs of the agentic process at the `INFO` level, making the creative process transparent without requiring `DEBUG` verbosity.
- **TTS Verbose Mode**: The `verbose` parameter for `Dia.generate()` is now set to `True` when the application's logging level is `INFO` or `DEBUG`.

### Fixed
- **Observability**: The `Final script after Director's Review (changes made):` log message now correctly appears when the Director's review removes all tags suggested by the Actor, ensuring that all modifications are logged.
- **CLI Usability**: The `output_path` argument is now optional when the `--dry-run` flag is used, improving the command-line experience.

## [0.1.5] - 2025-08-16

### Added
- **Enhanced Voice Generation**: Added Dia's high-quality voice cloning:
  * High-Fidelity Cloning: Users can now provide both an audio prompt and its matching transcript via new `--s{1|2}-transcript` flags for the most accurate voice cloning
- **Mandatory Seeding Policy**: Implemented a strict seeding policy for consistent voice generation:
  * When any speaker uses Pure TTS mode and the generation spans multiple audio blocks, a seed is mandatory
  * If no seed is provided, the application generates a secure random seed and logs it for reproducibility
  * The chosen seed is reset before every block generation to ensure voice consistency

## [0.1.4] - 2025-08-15

### Added
- **Persistent, Resumable Rehearsals:** Implemented `SqliteSaver` checkpointing for the agentic workflow. The process can now be safely interrupted and will automatically resume from the last completed line, saving time and compute on long transcripts.
- A comprehensive integration test to verify the "crash-and-resume" functionality of the new persistence layer.
- **Director's Final Cut:** Implemented a new, configurable review step in the agentic workflow to enforce verbal tag budgets. This includes two modes: a fast, rule-based "procedural" review and a higher-quality, context-aware "llm" review, with automatic fallback protection to ensure robustness.

### Changed
- **Agentic Workflow Engine:** Refactored the core agentic orchestrator from a procedural `for` loop to a formal LangGraph state machine. This significantly improves robustness, modularity, and enables the new stateful, resumable operations.
- Updated `CONFIGURATION.md` and `README.md` to document the new persistence features and explain the automatic in-memory database behavior during testing.

## [0.1.3] - 2025-08-14

### Fixed
- The actor's prompt was incorrectly indicating an internal maximum of only one change per moment instead of one change per _line_.
- The director's moment definition prompt was incorrectly asserting there was a previous moment even if there was not one.
- Optimized `_execute_full_moment()`
- Improves seed handling for determinstic output.

## [0.1.2] - 2025-08-12

### Added
- Implemented comprehensive `DEBUG` level logging for the agentic workflow. Developers can now enable `--verbosity DEBUG` to inspect the exact prompts sent to and raw responses received from the LLM by both the Director and Actor agents, providing full observability into the creative decision-making process.

### Fixed
- Fixed a regression in the Director agent's narrative analysis that caused it to treat entire transcripts as a single conversational moment. The agent now correctly identifies multiple, distinct moments (e.g., shifting from a business discussion to a personal one), leading to more contextually aware and higher-quality verbal tag placement.

## [0.1.1] - 2025-08-11

### Added
- Unified application logging and observability system
- CLI verbosity flags (-v/--verbose and --verbosity)
- Developer guide for observability features

### Changed
- Replaced all print() statements with proper logging calls
- Updated documentation to reflect new logging system
- Bumped version to 0.1.1

### Removed
- Direct print() statements throughout the codebase

## [0.1.0] - 2025-08-10

### Added
- Initial release with core podcast generation functionality
- Agentic workflow for verbal tag injection
- Dia TTS integration
- LiteLLM provider-agnostic LLM support

## [0.0.0] - 2025-08-08
- initial prototype
