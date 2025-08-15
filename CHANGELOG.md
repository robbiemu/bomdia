# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.3] - 2025-08-14

### Fixed
- The actor's prompt was incorrectly indicating an internal maximum of only one change per moment instead of one change per _line_.
- The director's moment definition prompt was incorrectly asserting there was a previous moment even if there was not one.
- Optimized `_execute_full_moment()`

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
