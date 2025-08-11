# Verbal Tag Injector Component

This component is responsible for intelligently adding verbal tags (like laughter, sighs, etc.) to a transcript to make it sound more natural when converted to audio.

## Architecture

The Verbal Tag Injector uses a two-agent system:

### Director Agent
The Director is the main orchestrator that:
- Performs an initial analysis of the entire script to generate a global summary
- Iterates through each line of the transcript
- For lines requiring nuance, prepares a detailed briefing packet for the Actor
- Reviews and audits the Actor's suggestions
- Enforces global rules (like tag quotas)
- Compiles the final enhanced script

### Actor Agent
The Actor is a specialized agent that:
- Receives a detailed briefing packet from the Director
- Performs a "take" on a single line of dialogue
- Returns a suggested version of the line with appropriate verbal tags

## Workflow

The process follows these steps:

1. **Pre-Production**: The Director reads the entire transcript and generates a global summary
2. **Rehearsal Loop**: For each line:
   - The Director checks if the line needs enhancement
   - If needed, the Director prepares a briefing packet
   - The Actor performs a "take" based on the packet
   - The Director reviews and integrates the suggestion
3. **Post-Production**: The final enhanced script is produced

## Configuration

The component uses prompts defined in `config/prompts.toml`:
- `global_summary_prompt`: Used by the Director for initial script analysis
- `unified_moment_analysis_prompt`: Used by the Director for line-by-line analysis
- `quota_exceeded_note`: Added to Director's notes when tag quota is reached
- `task_directive_template`: Template for the Actor's task directive

Pipeline parameters are configured in `config/app.toml`:
- `max_tag_rate`: Maximum rate of verbal tag insertion
- `context_window`: Context window size for verbal tag injection
