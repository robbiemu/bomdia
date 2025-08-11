# Configuration Guide

This guide covers all configuration options available in `config/app.toml` and `config/prompts.toml`, with detailed examples for different LLM providers.

## Table of Contents
- [Configuration Guide](#configuration-guide)
  - [Table of Contents](#table-of-contents)
  - [app.toml Configuration](#apptoml-configuration)
    - [Model Settings](#model-settings)
    - [Pipeline Settings](#pipeline-settings)
    - [Tags Configuration](#tags-configuration)
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

# Random seed for reproducible voice selection
seed = 42
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

# Note added when the tag quota has been exceeded
quota_exceeded_note = """
(Automated message: The production's budget for verbal tags has been met. Do not add any new parenthetical tags.)
"""
```

### Actor Agent Prompts

The Actor agent uses this prompt template to perform its interpretation of a line.

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
