# Configuration Guide

This guide covers all configuration options available in `config/app.toml` and `config/prompts.toml`, with detailed examples for different LLM providers.

## Table of Contents
- [Configuration Guide](#configuration-guide)
  - [Table of Contents](#table-of-contents)
  - [app.toml Configuration](#apptoml-configuration)
    - [Model Settings](#model-settings)
    - [Pipeline Settings](#pipeline-settings)
    - [Tags Configuration](#tags-configuration)
  - [Persistence and Checkpointing](#persistence-and-checkpointing)
    - [Checkpointing Behavior](#checkpointing-behavior)
    - [Resuming from Checkpoints](#resuming-from-checkpoints)
  - [Environment Variables](#environment-variables)
    - [LLM Configuration Override](#llm-configuration-override)
  - [Provider-Specific Examples](#provider-specific-examples)
    - [OpenAI](#openai)
    - [OpenRouter](#openrouter)
    - [Mistral AI](#mistral-ai)
    - [OpenAI-Compatible Endpoints](#openai-compatible-endpoints)
  - [prompts.toml Configuration](#promptstoml-configuration)
    - [Director Agent Prompts](#director-agent-prompts)
    - [Actor Agent Prompts](#actor-agent-prompts)
  - [Troubleshooting Configuration Issues](#troubleshooting-configuration-issues)

## app.toml Configuration

### Model Settings

The `[model]` section configures the text-to-speech and LLM components.

```toml
[model]
# Dia TTS model configuration
dia_checkpoint = "nari-labs/Dia-1.6B-0626"
dia_checkpoint_revision = "main"
# Compute precision for Dia TTS model (options: "float16", "float32", "bfloat16")
dia_compute_dtype = "float16"
# Optional: Specify the compute device. Options: "auto", "cuda", "mps", "cpu".
# Defaults to "auto" for automatic detection.
# device = "auto"

# LiteLLM model specification
# Format: "provider/model-name" or "provider/model-name@provider"
llm_spec = "openai/gpt-4o-mini"

# Model parameters for LLM
[model.parameters]
temperature = 0.5
max_tokens = 150
top_p = 0.9
frequency_penalty = 0.0
presence_penalty = 0.0

# Dia TTS generation parameters
[model.dia_generate_params]
max_tokens = 3072
cfg_scale = 3.0
temperature = 1.2
top_p = 0.95
cfg_filter_top_k = 45
use_cfg_filter = false

### Device Selection

You can explicitly set the compute device for the TTS model.

```toml
[model]
# Options: "auto", "cuda", "mps", "cpu"
device = "cpu"
```

The default is `"auto"`, which will prioritize `cuda`, then `mps`, then `cpu`.

You can also override this setting with the `BOMDIA_DEVICE` environment variable:

```bash
export BOMDIA_DEVICE="cpu"
```

### Pipeline Settings

The `[pipeline]` section controls transcript processing behavior.

```toml
[pipeline]
# Context window size for verbal tag injection
context_window = 2

# Placeholder used when merging consecutive lines
pause_placeholder = "[insert-verbal-tag-for-pause]"

# Maximum rate of verbal tag insertion (0.0-1.0)
max_tag_rate = 0.15

# Average words per second for audio estimation
avg_wps = 2.5

# Maximum tokens for Dia generation
max_new_tokens_cap = 1600

# Default random seed for reproducible voice selection
# Note: When using pure TTS mode for speakers in multi-block transcripts,
# a seed is mandatory and will be auto-generated if not provided
seed = 42
```

### Director Agent Rate Control

The `[director_agent.rate_control]` section configures the token bucket algorithm for the Director Agent's tagging logic.

```toml
[director_agent.rate_control]
# The target ratio of lines that should receive a new verbal tag (e.g., 0.10 for 10%).
target_tag_rate = 0.10

# The maximum number of "burst" tags the agent can inject above its target rate.
# This allows it to handle emotionally dense scenes.
tag_burst_allowance = 3
```

### Tags Configuration

The `[tags]` section defines verbal tags and line combiners. These are defined by the tts model and should not be modified (unless you are changing the tts model in the code, in which case you already know what you are doing).

```toml
[tags]
# Verbal tags that can be automatically inserted
verbal_tags = [
    "(laughs)", "(clears throat)", "(sighs)", ...
]
```

However, line combiners are interpolated into lines that are so long that they must be split, in order to help smooth over the combination of audio chunks that must be rendered separately. Thse are user-configurable.

```toml
[tags]
# Line combiners used when merging consecutive lines
line_combiners = [
    "…um,",
    "- uh -",
    "— hmm —",
]
```

## Persistence and Checkpointing

The `[persistence]` section configures the SQLite database used for checkpointing the agentic workflow state. This allows the system to resume from where it left off in case of interruption.

```toml
[persistence]
# Path to the SQLite database file for checkpointing
rehearsal_checkpoint_path = "rehearsal_checkpoints.sqlite"
```

### Checkpointing Behavior

When running tests, the system automatically uses an in-memory SQLite database (`:memory:`) to prevent polluting the repository with database files. This behavior is controlled by the `PYTEST_CURRENT_TEST` environment variable.

In production, the system uses the configured file path. If the file doesn't exist, it will be created automatically.

### Resuming from Checkpoints

The agentic workflow automatically detects and resumes from existing checkpoints. Each execution is associated with a unique thread ID, which is used to identify the checkpoint state.

## Environment Variables

### LLM Configuration Override

You can override configuration file settings using environment variables:

```bash
# Override LLM model
export LLM_SPEC="openai/gpt-4-turbo"

# Override model parameters
export TEMPERATURE="0.8"
export MAX_TOKENS="300"

# Override pipeline settings
export CONTEXT_WINDOW="3"
export MAX_TAG_RATE="0.20"
```

## Provider-Specific Examples

### OpenAI

For detailed OpenAI configuration, see the [LiteLLM OpenAI documentation](https://docs.litellm.ai/docs/providers/openai).

```toml
[model]
llm_spec = "openai/gpt-4o-mini"

[model.parameters]
temperature = 0.7
max_tokens = 200
top_p = 0.9
frequency_penalty = 0.0
presence_penalty = 0.0
```

**Required Environment Variables:**
```bash
export OPENAI_API_KEY="sk-your-openai-key-here"
```

### OpenRouter

For detailed OpenRouter configuration, see the [LiteLLM OpenRouter documentation](https://docs.litellm.ai/docs/providers/openrouter).

```toml
[model]
llm_spec = "openrouter/openai/gpt-4o-mini"

[model.parameters]
temperature = 0.5
max_tokens = 150
```

**Required Environment Variables:**
```bash
export OPENROUTER_API_KEY="your-openrouter-key"
```

### Mistral AI

For detailed Mistral AI configuration, see the [LiteLLM Mistral documentation](https://docs.litellm.ai/docs/providers/mistral).

```toml
[model]
llm_spec = "mistral/mistral-large-latest"

[model.parameters]
temperature = 0.6
max_tokens = 150
top_p = 0.95
```

**Required Environment Variables:**
```bash
export MISTRAL_API_KEY="your-mistral-api-key"
```

### OpenAI-Compatible Endpoints

For detailed OpenAI-compatible endpoint configuration, see the [LiteLLM OpenAI Compatible documentation](https://docs.litellm.ai/docs/providers/openai_compatible).

```toml
[model]
llm_spec = "openai/your-model-name"

[model.parameters]
temperature = 0.7
max_tokens = 200
```

**Required Environment Variables:**
```bash
export OPENAI_API_BASE="http://localhost:1234/v1"
export OPENAI_API_KEY="not-needed-for-local"
```

## prompts.toml Configuration

These are prompts used in the agentic process of reformulating the transcript into a dia-tts-specific format. They may be tuned to elicit the best performance from your configured model.

### Director Agent Prompts

The Director agent uses these prompts to analyze the script and guide the Actor.

```toml
[director_agent]
# Prompt for the Director's initial, one-time analysis of the entire script.
global_summary_prompt = """
You are a script analyst. Read the entire following transcript and provide a concise summary covering three key areas. This summary will be used to guide another AI in performing the dialogue, so clarity and insight are crucial.

1.  **Overall Topic:** What is the main subject of the conversation?
2.  **Speaker Relationship:** Describe the dynamic between the speakers (e.g., friendly colleagues, confrontational rivals, interviewer and subject).
3.  **Emotional Arc:** Describe the flow of emotion from the beginning to the end of the conversation (e.g., "Starts with lighthearted banter, moves to a serious disagreement, and ends with a reluctant compromise.").

Here is the transcript:
---
{transcript_text}
---
"""

# Unified prompt for the Director to get a quick analysis of a specific moment.
unified_moment_analysis_prompt = """
You are an acting coach providing a quick note. Analyze this brief exchange, focusing ONLY on the speaker of the 'Current Line'.

Return a JSON object with two keys:
- "moment_summary": A third-person analysis of the speaker's emotional state and intention.
- "directors_note": A first-person, actionable command for the Actor.

**Exchange:**
{local_context}

**Current Line:**
{current_line}
"""

# Prompt for the Director to define narrative moments in the script.
moment_definition_prompt = """
You are a script director, preparing your notes for the next performance.

The Previous Moment:
You just completed a moment described as: "{last_moment_summary}".
It concluded on line {last_moment_end_line} with the finalized performance: "{last_finalized_line_text}".

Your Task:
Now, analyze the upcoming script to define the next `current moment`. A `current moment` is a continuous, self-contained beat with a consistent emotional tone and narrative purpose.

A moment is defined by a consistent and unbroken:
*   Topic: The characters are talking about the same immediate subject.
*   Intention: A character is trying to achieve a single, specific goal.
*   Emotional Tone: The underlying feeling is consistent.

A moment ends when one of these characteristics clearly shifts.

Examples:
1. Single, Simple Moment:
[S1] So, what did you think of the movie?
[S2] Honestly, I thought it was a bit slow in the middle.
[S1] Really? I loved the pacing. It felt deliberate.
Start Line: 0, End Line: 2

2. Two Distinct Moments:
# Moment A Starts
[S1] Okay, so the quarterly report is due Friday.
[S2] Right. I've already finished the sales figures section.
[S1] Perfect, I'll handle the marketing summary.
# Moment A Ends, Moment B Starts
[S2] Oh, by the way, did you see that email from HR about the new policy?
[S1] No, what did it say?
# Moment B Ends
Moment A: Start Line: 0, End Line: 2
Moment B: Start Line: 3, End Line: 4

3. The "Pivot Line"
A single line can be the end of one moment and the start of another. This is the most complex case.
# Moment A (Party) Starts
[S1] ...and that's why we're all so happy for you!
[S2] To the happy couple!
[EVERYONE] Happy birthday to you!  <-- PIVOT LINE
# Moment A Ends, Moment B (Sadness) Starts
[S3] (Hearing the song from a distance) That was her favorite song.
[S4] I know. Let's get out of here.
# Moment B Ends
Moment A: Start Line: 0, End Line: 2
Moment B: Start Line: 2, End Line: 4

Upcoming Script (with line numbers):
---
{forward_script_slice_text}
---

Your Direction:
Please respond with a single JSON object that provides your complete direction for this next moment.

{
  "moment_summary": "Concise description of what is happening in this new moment. What is the core emotion and the characters' primary intentions?",
  "directors_notes": "Actionable notes for the actors. What should they be feeling or trying to achieve during this moment?",
  "start_line": {line_number},
  "end_line": {line_number + 1}
}

Respond with ONLY the JSON object, no other text.
"""

# Note added when the tag quota has been exceeded
quota_exceeded_note = """
(Automated message: The production's budget for verbal tags has been met. Do not add any new parenthetical tags.)
"""
```

### Actor Agent Prompts

The Actor agent uses these prompt templates to perform its interpretation of lines.

```toml
[actor_agent]
# Template for the Actor's main task. This will be formatted with context by the Director.
task_directive_template = """
You are a voice actor performing a line from a script. Your performance should be natural and enhance the written dialogue.

**Context:**
- **Global Summary:** {global_summary}
- **Local Context (The lines immediately surrounding yours):**
{local_context}
- **Director's Interpretation of the Moment:** {moment_summary}
- **Director's Notes:** {directors_notes}

**Your Task:**
Perform ONLY the following line:
**Current Line:** {current_line}

**Performance Rules:**
1.  **Regarding `[insert-verbal-tag-for-pause]`:** This placeholder marks a technical break in a single continuous thought. Your job is to bridge this gap naturally. Replace it with an appropriate verbal hesitation (e.g., "…um,", "--- hmm ---", "(sighs)"). If no verbalization feels right, you MUST replace it with a single space to connect the parts.
2.  **Regarding New Verbal Tags:** You may add ONE short verbal tag (e.g., "(laughs)", "(gasps)") to the beginning of the line, but ONLY if it is strongly motivated by the emotional context. Do not add tags arbitrarily.
3.  **Output:** Return ONLY the single, modified line of dialogue. Do not include the speaker tag (e.g., `[S1]`) or any commentary.
"""

# Template for the Actor's moment-based task.
moment_task_directive_template = """
You are a voice actor performing lines from a script. Your performance should be natural and enhance the written dialogue.

**Context:**
- **Global Summary:** {global_summary}

**Your Task:**
Perform the following lines as a cohesive moment:
{moment_text}

**Performance Rules:**
1.  **Regarding `[insert-verbal-tag-for-pause]`:** This placeholder marks a technical break in a single continuous thought. Your job is to bridge this gap naturally. Replace it with an appropriate verbal hesitation (e.g., "…um,", "--- hmm ---", "(sighs)"). If no verbalization feels right, you MUST replace it with a single space to connect the parts.
2.  **Regarding New Verbal Tags:** You may add ONE short verbal tag (e.g., "(laughs)", "(gasps)") to the beginning of a line, but ONLY if it is strongly motivated by the emotional context. Do not add tags arbitrarily.
3.  **Token Budget:** You have approximately {token_budget:.1f} tokens available for your performance.
{constraints_text}

**Output Format:**
Respond with a JSON object that maps each line's global line number to its performed text:
{
  "line_0": "Performed text for line 0",
  "line_1": "Performed text for line 1"
}

Make sure to return ONLY the JSON object, with no additional text or markdown formatting.
"""
```

## Troubleshooting Configuration Issues

1. **Check current configuration:**
   ```python
   from shared.config import config
   print(f"LLM: {config.LLM_SPEC}")
   print(f"Parameters: {config.LLM_PARAMETERS}")
   ```

2. **Verify environment variables:**
   ```bash
   env | grep -E "(LLM|API|OPENAI|MISTRAL|OPENROUTER)"
   ```

## Logging and Observability

The application uses Python's standard `logging` module for all output. There are two layers of observability:

1. **Default Logging System**: All application output is now handled through a configurable logging system with controllable verbosity via CLI flags:
   - `-v`, `--verbose`: Sets logging level to INFO
   - `--verbosity {DEBUG,INFO,WARNING,ERROR}`: Sets a specific logging level

2. **LangSmith Tracing** (Optional): For developers who need detailed tracing information, LangSmith integration can be enabled by setting the appropriate environment variables:
   - `LANGCHAIN_TRACING_V2=true`
   - `LANGCHAIN_ENDPOINT=https://api.smith.langchain.com`
   - `LANGCHAIN_API_KEY=your-langsmith-api-key`
   - `LANGCHAIN_PROJECT=your-project-name`

The LangSmith integration requires no application-level configuration and is managed solely by these environment variables. The application runs without error or warning if they are not set.
