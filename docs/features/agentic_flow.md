### The Director and the Actor: An Agentic Rehearsal Workflow

This workflow uses a two-agent system to intelligently enhance a transcript.

**The Agents:**

*   **The Director (Main Agent):** The orchestrator and final decision-maker. It manages the entire script, maintains global state (like the tag rate), and directs the Actor. It is responsible for producing the final, compiled script.
*   **The Actor (Sub-Agent):** A specialized, creative agent. Its job is to perform "takes" on narrative moments, given a rich set of directions from the Director. It is stateless and focused purely on interpretation.

---

### The Rehearsal Process: A Step-by-Step Breakdown

#### **Phase 1: Pre-Production (The Director's Prep)**

The Director agent is instantiated with the pre-processed transcript.

1.  **Script Read-Through:** The Director performs a single pass over the entire transcript to generate a **`global_summary`**. This summary contains its high-level understanding of the script's topic, speaker dynamics, and the **overall emotional arc**.
2.  **Production Setup:** The Director initializes its state variables for the "shoot":
    *   `final_script = []`
    *   `tags_injected = 0`
    *   `max_tags_allowed = total_lines * 0.15`

#### **Phase 2: The Rehearsal Loop (Moment-by-Moment Direction)**

The Director iterates through the script line by line, discovering and defining narrative moments. For each line:

**A. Moment Discovery and Definition**

The Director analyzes the script to identify narrative moments based on consistent topic, intention, and emotional tone. A moment is a continuous, self-contained beat that makes sense as a unit.

**B. Moment Completion and Delegation**

When a moment is complete (all lines in the moment have been encountered), the Director prepares a comprehensive briefing packet for the Actor:

*   **`moment_text`**: All lines in the completed moment.
*   **`global_summary`**: The Director's high-level analysis of the entire script's emotional arc.
*   **`moment_summary`**: The Director's specific interpretation of the current moment.
*   **`directors_notes`**: Actionable notes for the Actor about how to perform this moment.
*   **`token_budget`**: The number of tokens the Actor can use for this performance.
*   **`constraints`**: Any special constraints for specific lines (e.g., pivot lines that are already finalized).

**C. The Actor's "Take" (Delegation)**

The Director sends this complete packet to the Actor agent, which performs its interpretation of the entire moment and returns its suggested lines.

**D. The Director's Review and Final Cut (Audit & Integration)**

The Director receives the Actor's "take" and, as the final authority, decides what makes it into the film.

1.  **Review the Take:** The Director compares the Actor's suggestions to the original lines.
2.  **Enforce the Rules:** The Director programmatically checks the suggestions against its global rules.
    *   If new tags were added, it checks its tag counter. If the quota is exceeded, the Director may strip some tags from the Actor's suggestions.
3.  **Compile the Script:** The Director appends the final, approved versions of the lines to its `final_script` list.

#### **Phase 3: Post-Production**

Once the loop is complete, the `final_script` contains the fully enhanced, rule-compliant transcript, ready for the audio rendering pipeline. The Director's job is done.
