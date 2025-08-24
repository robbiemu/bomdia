# Configuration Guide

This guide covers all configuration options available in `config/app.toml` and `config/prompts.toml`, with detailed examples for different LLM providers.

## Table of Contents
- [Configuration Guide](#configuration-guide)
  - [Table of Contents](#table-of-contents)
  - [app.toml Configuration](#apptoml-configuration)
    - [Model Settings](#model-settings)
      - [dia parameters](#dia-parameters)
        - [Temperature](#temperature)
        - [Top p and top k](#top-p-and-top-k)
    - [Device Selection](#device-selection)
    - [Pipeline Settings](#pipeline-settings)
    - [Director Agent Rate Control](#director-agent-rate-control)
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
  - [Logging and Observability](#logging-and-observability)

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
llm_spec = "cerebras/llama-4-scout-17b-16e-instruct"

# Model parameters for LLM
[model.parameters]
temperature = 1
max_tokens = 150
top_p = 0.9
frequency_penalty = 0.0
presence_penalty = 0.0

# Dia TTS generation parameters
[model.dia_generate_params]
max_tokens = 3072
cfg_scale = 4.5
temperature = 1.2
top_p = 0.95
cfg_filter_top_k = 45
use_cfg_filter = false
````

#### dia parameters

Here are some definitions from the dia tts model.py source code:

-  **max_tokens**: The maximum number of audio tokens to generate per prompt.Defaults to the model´s configured audio length if None.
-  **cfg_scale**: The scale factor for classifier-free guidance (CFG). Higher values lead to stronger guidance towards the text prompt.
-  **temperature**: The temperature for sampling. Higher values increase randomness.
-  **top_p**: The cumulative probability threshold for nucleus (top-p) sampling.
-  **use_torch_compile**: Whether to compile the generation steps using torch.compile. Can significantly speed up generation after the initial compilation overhead. Defaults to False.
- **cfg_filter_top_k**: The number of top logits to consider during CFG filtering.

What follows is speculation about these parameters in dia TTS.

##### Temperature
The way **temperature** works appears to be different than you might expect. A higher temperature primarily effects rate of speech and the incidence of pauses (these decrease with increasing temperature). If you are having trouble with voices changing, you may find it adheres better to the voice when this is set higher (say > 1.6).

##### Top p and top k
These work in an unusual way as well. A low top_p will not help voices adhere at the cost of monotony. Instead, a lower top_p increases the chance that silence is selected.

The dia project also upped their top_k from 30 to 45.

However, the basic ideas are all familiar. Higher temperature says "weight the choices by some factor" so the best ones stand out more, a higher top p says "try to sample more possible next tokens", and a higher top k says "in case we get too many samples, only consider this many".


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
max_tag_rate = 0.25

# Average words per second for audio estimation
avg_wps = 2.875

# Default random seed for reproducible voice selection
# Note: When using pure TTS mode for speakers in multi-block transcripts,
# a seed is mandatory and will be auto-generated if not provided
seed = 42

# Whether to enable fully deterministic mode for debugging
# fully_deterministic = false

# Minimum duration for audio chunks in seconds
# min_chunk_duration = 5.0

# Maximum duration for audio chunks in seconds
# max_chunk_duration = 10.0

# Whether to generate synthetic voice prompts for unprompted speakers (default: True)
generate_synthetic_prompts = false
```

### Director Agent Rate Control

The `[director_agent.rate_control]` section configures the token bucket algorithm for the Director Agent's tagging logic.

```toml
[director_agent.rate_control]
# The target ratio of lines that should receive a new verbal tag (e.g., 0.10 for 10%).
target_tag_rate = 0.16

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
# System prompt for Director role – used across all Director tasks
system_prompt = """Reasoning: high
You are a script director. You analyze scripts, define narrative moments, and review actor performances to create the final approved version. You respond with structured formats when specified and provide clear direction.
"""

# Prompt for the Director's initial, one-time analysis of the entire script.
global_summary_prompt = """
Working as the script analyst, read the entire following transcript and provide a concise summary covering three key areas. This summary will be used to guide another AI in performing the dialogue, so clarity and insight are crucial.

1.  **Overall Topic:** What is the main subject of the conversation?
2.  **Speaker Relationship:** Describe the dynamic between the speakers (e.g., friendly colleagues, confrontational rivals, interviewer and subject).
3.  **Emotional Arc:** Describe the flow of emotion from the beginning to the end of the conversation (e.g., "Starts with lighthearted banter, moves to a serious disagreement, and ends with a reluctant compromise.").

Here is the transcript:
---
{transcript_text}
---
"""

# Prompt to define narrative moments.
moment_definition_prompt = """
You are preparing your notes for the next performance.
{previous_moment_segment}

Your Task:
Analyze the upcoming script to define the **next `current moment`**. A `current moment` is a continuous, self-contained beat with a consistent emotional tone and narrative purpose. It starts at the first unassigned line and ends when the topic, intention, or emotional tone clearly shifts.

Upcoming Script (with global line numbers):
---
{forward_script_slice_text}
---

Your Direction:
Based on the script, identify the boundaries of the very next moment. Respond with a single JSON object that provides your complete direction.

{
  "moment_summary": "Concise description of what is happening in this new moment. What is the core emotion and the characters' primary intentions?",
  "directors_notes": "Actionable notes for the actors. What should they be feeling or trying to achieve during this moment?",
  "start_line": {line_number},
  "end_line": <The line number where this moment naturally concludes>
}

Respond with ONLY the JSON object, no other text.
"""

previous_moment_template = """
The Previous Moment:
You just completed a moment described as: "{last_moment_summary}".
It concluded on line {last_moment_end_line} with the finalized performance: "{last_finalized_line_text}".
"""

quota_exceeded_note = """
(Automated message: The production's budget for verbal tags has been met. Do not add any new parenthetical tags.)
"""

# Prompt for the Director to review and finalize an Actor's performance for a moment.
director_review_prompt = """
Be meticulous in reviewing this take from a voice actor and prepare the final approved version of the script for the specific narrative moment, ensuring it adheres to a strict budget for new verbal tags. Slight wording changes may be permitted at your discretion, but omitted phrases or sentences must be restored.

**CONTEXT**

*   **Global Summary:** {global_summary}
*   **Previous Moment Performance:** This is the finalized script from the moment that just concluded. It was about "{last_moment_summary}".
    ---
    {previous_moment_performance_text}
    ---
*   **Current Moment - Original Script:** This is the script before the Actor's performance.
    ---
    {original_script_text}
    ---
*   **Actor's Performance (Your 'Take' to Review):** This is the version submitted by the Actor, which may include new verbal tags.
    ---
    {actor_performance_text}
    ---

**YOUR DIRECTIVE**

You have a budget to add a maximum of **{tag_budget}** new verbal tags for this entire moment.

1.  **Review the Performance:** Compare the "Original Script" to the "Actor's Performance." Identify all the new parenthetical verbal tags (e.g., `(laughs)`, `(sighs)`) the Actor added.
2.  **Enforce the Budget:**
    *   If the number of new tags is **over budget**, you MUST revert some of the Actor's changes back to the original text until the budget is met. Preserve the most impactful and contextually motivated additions.
    *   If the performance is **within budget**, you should approve it as-is, unless a change feels genuinely out of place or unmotivated.
3.  **Editing Constraint:** Your editing is strictly limited. To construct the final script, you must decide for each line whether to **keep** the Actor's version or **revert** some or all of it to the original text, following these rules:
    *   The placeholder `[insert-verbal-tag-for-pause]` is a technical directive and **must not** appear in the final text. If the Actor replaced it, you must keep their replacement or remove it. If the Actor's take contains a placeholder still, you must remove it.
    *   If any lines or phrases were omitted, you must restore them (the actors may have smaller omissions or changes that may be kept at your discretion).
    *   You **cannot** write new dialogue, add your own creative tags, or modify the text in any other way.

**OUTPUT FORMAT**

Respond with ONLY a JSON object that maps each line's global line number to its final, approved text. Return all the lines of the take after your edits.

Example:
{
  "line_{start_line}": "Final text for the first line in the moment.",
  "line_{start_line_plus_1}": "Final text for the second line, which may have been reverted to its original version."
}

+**IMPORTANT:** The text you return for each line should NOT include the speaker name (e.g., `[{sample_name}]`). Return only the dialogue.

Do not include any other text, explanations, or markdown formatting in your response.
"""
```

### Actor Agent Prompts

The Actor agent uses these prompt templates to perform its interpretation of lines.

```toml
[actor_agent]
# System prompt for Actor role – used across all Actor tasks
system_prompt = """Reasoning: high
You are a voice actor performing lines from a script. Your performance should be natural and enhance the written dialogue. Respond in structured JSON format.
"""

# Template for the Actor's moment-based task.
moment_task_directive_template = """
Your performance should be natural and enhance the written dialogue.

**Context:**
- **Global Summary:** {global_summary}

**Your Task:**
Perform the following lines as a cohesive moment:
{moment_text}

**Available Verbalizations:**
- **For emotional expression:** You may add tags from the following list: {available_verbal_tags}
- **For pauses/hesitations:** When replacing `[insert-verbal-tag-for-pause]`, you must use one of the emotional expressions, or one following: {available_line_combiners}

**Performance Rules:**
1.  **Regarding `[insert-verbal-tag-for-pause]`:** This placeholder marks a technical break in a single continuous thought. Your job is to bridge this gap naturally by replacing it with an appropriate option from the list above. If no verbalization feels right, you MUST replace it with a single space to connect the parts.
2.  **Regarding New Verbal Tags:**
    *   You have a gentle budget of up to **{token_budget:.0f}** new verbal tags for the entire moment.
    *   Sprinkle them wherever they make the dialogue breathe—no need to use the full allowance if the scene feels fine without them. Even serious or technical dialogue can include pauses, sighs, or chuckles that reveal the speaker's personality.
    *   At most one new tag per line, please.

**Output Format:**
Respond with a JSON object mapping each line to its performed text, using the line's number for the key as shown in the example:
{
  "line_{start_line}": "Performed text for the first line in the moment",
  "line_{start_line_plus_1}": "Performed text for the second line in the moment"
}

+**IMPORTANT:** The text you return for each line should NOT include the speaker name (e.g., `[{sample_name}]`). Return only the dialogue.

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
