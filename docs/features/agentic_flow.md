### The Director and the Actor: An Agentic Rehearsal Workflow

This workflow uses a two-agent system to intelligently enhance a transcript.

**The Agents:**

*   **The Director (Main Agent):** The orchestrator and final decision-maker. It manages the entire script, maintains global state (like the tag rate), and directs the Actor. It is responsible for producing the final, compiled script.
*   **The Actor (Sub-Agent):** A specialized, creative agent. Its only job is to perform a "take" on a single line of dialogue, given a rich set of directions from the Director. It is stateless and focused purely on interpretation.

---

### The Rehearsal Process: A Step-by-Step Breakdown

#### **Phase 1: Pre-Production (The Director's Prep)**

The Director agent is instantiated with the pre-processed transcript.

1.  **Script Read-Through:** The Director performs a single pass over the entire transcript to generate a **`global_summary`**. This summary contains its high-level understanding of the script's topic, speaker dynamics, and the **overall emotional arc**.
2.  **Production Setup:** The Director initializes its state variables for the "shoot":
    *   `final_script = []`
    *   `tags_injected = 0`
    *   `max_tags_allowed = total_lines * 0.15`

#### **Phase 2: The Rehearsal Loop (Line-by-Line Direction)**

The Director iterates through the script line by line. For each `current_line`:

**A. The Director's Triage (Optimization)**

The Director first performs a quick, programmatic check on the `current_line`.

*   **Is the line simple and non-emotional?** (e.g., `"[S2] Yes."`, `"[S1] Okay."`) or does it not contain a pause placeholder?
*   If yes, the Director determines that the Actor's input is not needed. It performs a **"no-op"**: it appends the line directly to the `final_script` and moves to the next line. This saves significant time and cost by skipping unnecessary LLM calls.

**B. Preparing the "Sides" (The Briefing Packet for the Actor)**

If the line requires nuance, the Director prepares the Actor's script pages, or "sides." This is a precisely crafted prompt context.

*   **`current_line`**: The line to be performed.
*   **`local_context`**: The Director provides the immediately preceding and subsequent lines to give the Actor a sense of the immediate moment.
*   **`global_summary`**: The Director's high-level analysis of the entire script's emotional arc.
*   **`moment_summary`**: The Director's specific interpretation of the current beat. This combines the previous `moment_summary` and `speaker_headspace`. (e.g., *"This is a moment of confused hesitation. The speaker is trying to reconcile what they just heard with what they were about to say."*)
*   **`task_directive`**: A configurable prompt template that outlines the Actor's core task. This is where we explain the `[insert-verbal-tag-for-pause]` tag's purpose.
    > *"Perform this line. If you see `[insert-verbal-tag-for-pause]`, it's a technical break; bridge it with a natural hesitation or phrase. If nothing fits, use a simple space. You may also add one verbal tag at the start if the emotion strongly calls for it. Return only your performed version of the line."*
*   **`director's_notes`**: Dynamic, real-time feedback based on the Director's state.
    *   If `tags_injected` is getting high: *"NOTE: We're using too many tags. Be very subtle. Prioritize the placeholder and only add a new tag if absolutely essential."*
    *   If the Actor's recent suggestions have been repetitive: *"NOTE: Try to vary your choice of hesitations. Avoid using '…um,' again."*

**C. The Actor's "Take" (Delegation)**

The Director sends this complete packet to the Actor agent, which performs its interpretation and returns its suggested line.

**D. The Director's Review and Final Cut (Audit & Integration)**

The Director receives the Actor's "take" and, as the final authority, decides what makes it into the film.

1.  **Review the Take:** The Director compares the Actor's suggestion to the original line.
2.  **Enforce the Rules:** The Director programmatically checks the suggestion against its global rules.
    *   If a new tag was added, it checks its tag counter. If the quota is exceeded, the Director **strips the new tag** from the Actor's suggestion. It keeps any valid edits, like the replacement of the pause placeholder.
3.  **Compile the Script:** The Director appends the final, approved version of the line to its `final_script` list. **This is a simple programmatic action (e.g., `final_script.append(line)`), not another LLM call.**

#### **Phase 3: Post-Production**

Once the loop is complete, the `final_script` contains the fully enhanced, rule-compliant transcript, ready for the audio rendering pipeline. The Director's job is done.
