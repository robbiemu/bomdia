import json
import math
import random
import re
from typing import Dict, List

from shared.config import config
from shared.llm_invoker import LiteLLMInvoker
from shared.logging import get_logger
from src.components.verbal_tag_injector.actor import get_actor_suggestion

# Initialize logger
logger = get_logger(__name__)


class Director:
    def __init__(self, transcript: List[Dict]):
        self.transcript = transcript
        self.llm_invoker = LiteLLMInvoker(
            model=config.LLM_SPEC, **config.LLM_PARAMETERS
        )
        self.final_script: List[Dict] = []
        self.tags_injected = 0
        # Calculate the total budget of NEW tags we can inject (deterministic approach)
        self.new_tag_budget = math.floor(len(transcript) * config.MAX_TAG_RATE)
        logger.info(
            f"Director initialized with a budget of {self.new_tag_budget} "
            f"new verbal tags."
        )
        self.global_summary = self._generate_global_summary()

    def _generate_global_summary(self) -> str:
        """
        Generates a high-level summary of the transcript.
        """
        # The generation itself is a background detail. The result is what's important.
        logger.debug("Generating global summary...")
        transcript_text = "\n".join([line["text"] for line in self.transcript])
        prompt = config.director_agent["global_summary_prompt"].format(
            transcript_text=transcript_text
        )

        messages = [{"role": "user", "content": prompt}]
        response = self.llm_invoker.invoke(messages)
        # The summary itself is key context, so log it at INFO.
        logger.info(f"Global Summary Generated: {response.content}")
        return response.content

    def run_rehearsal(self) -> List[Dict]:
        """
        Main method to execute the director-actor workflow.
        """
        logger.info(f"Starting rehearsal process for {len(self.transcript)} lines...")
        for i, line in enumerate(self.transcript):
            # --- New, Correct Triage Logic ---
            is_placeholder_candidate = config.PAUSE_PLACEHOLDER in line["text"]

            # We are a candidate for a new tag if our budget isn't zero.
            # We can add a random element to not try on every single line.
            can_add_new_tag = (
                self.tags_injected < self.new_tag_budget
                and random.random() < 0.5  # nosec B311
            )  # e.g., 50% chance to try

            if not is_placeholder_candidate and not can_add_new_tag:
                # This line is not a candidate for either type of enhancement. Skip it.
                logger.debug(f"Line {i+1}: Skipping (not a candidate for enhancement).")
                self.final_script.append(line)
                continue

            # If we reach here, the line IS a candidate.
            # Proceed with the full Actor workflow.
            logger.info(f"--- Processing Line {i+1}: '{line['text']}' ---")

            local_context = "\n".join(
                [
                    line["text"]
                    for line in self.transcript[
                        max(0, i - 3) : min(len(self.transcript), i + 4)
                    ]
                ]
            )

            # The unified analysis call is a background action.
            logger.debug("Requesting unified moment analysis from LLM...")
            unified_prompt = config.director_agent[
                "unified_moment_analysis_prompt"
            ].format(local_context=local_context, current_line=line["text"])
            messages = [{"role": "user", "content": unified_prompt}]
            response_content = self.llm_invoker.invoke(messages).content

            try:
                # Clean up the response content by removing markdown code blocks
                # if present
                cleaned_response = response_content.strip()
                if cleaned_response.startswith("```json"):
                    cleaned_response = cleaned_response[7:]  # Remove ```json
                if cleaned_response.startswith("```"):
                    cleaned_response = cleaned_response[3:]  # Remove ```
                if cleaned_response.endswith("```"):
                    cleaned_response = cleaned_response[:-3]  # Remove ```
                cleaned_response = cleaned_response.strip()

                analysis = json.loads(cleaned_response)
                moment_summary = analysis.get("moment_summary", "")
                directors_note = analysis.get("directors_note", "")
            except json.JSONDecodeError:
                logger.debug(f"Failed to parse JSON response: {response_content}")
                moment_summary = ""
                directors_note = ""

            # The RESULTS of that analysis are the story. This is INFO.
            logger.info(f"  [Director's Analysis] Moment Summary: {moment_summary}")
            logger.info(f"  [Director's Analysis] Director's Note: {directors_note}")

            # In the new budget system, we don't need to append quota exceeded note here
            # The check happens at the final cut stage

            briefing_packet = {
                "task_directive_template": config.actor_agent[
                    "task_directive_template"
                ],
                "global_summary": self.global_summary,
                "local_context": local_context,
                "moment_summary": moment_summary,
                "directors_notes": directors_note,
                "current_line": line["text"],
            }

            # The actor call is a background action.
            logger.debug("Delegating to Actor for creative suggestion...")
            suggestion = get_actor_suggestion(briefing_packet, self.llm_invoker)

            # The Actor's performance is a key part of the story. This is INFO.
            logger.info(f"  [Actor's Performance] Suggestion: '{suggestion}'")

            # Check if a new tag was added (not just replacing the placeholder)
            original_has_tag = bool(re.search(r"\([^)]+\)", line["text"]))
            suggestion_has_new_tag = bool(re.search(r"\([^)]+\)", suggestion))
            new_tag_added = suggestion_has_new_tag and not original_has_tag

            if new_tag_added:
                # A new tag was added, check budget
                if self.tags_injected < self.new_tag_budget:
                    logger.info(
                        f"  [Director's Final Cut] New tag approved. "
                        f"Budget remaining: "
                        f"{self.new_tag_budget - self.tags_injected - 1}"
                    )
                    self.final_script.append(
                        {
                            "speaker": line["speaker"],
                            "text": suggestion,
                        }
                    )
                    self.tags_injected += 1
                else:
                    # Strip the new tag using regex
                    suggestion = re.sub(r"\(.*?\)", "", suggestion)
                    logger.info(
                        "  [Director's Final Cut] New tag REJECTED "
                        "(budget exhausted). Stripping tag."
                    )
                    self.final_script.append(
                        {
                            "speaker": line["speaker"],
                            "text": suggestion,
                        }
                    )
            else:
                # No new tag was added, just append the suggestion
                logger.info("  [Director's Final Cut] Suggestion approved.")
                self.final_script.append(
                    {
                        "speaker": line["speaker"],
                        "text": suggestion,
                    }
                )

        logger.info("--- Rehearsal process complete. ---")
        return self.final_script
