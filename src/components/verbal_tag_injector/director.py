import json
import re
from typing import Dict, List

from shared.config import config
from shared.llm_invoker import LiteLLMInvoker
from src.components.verbal_tag_injector.actor import get_actor_suggestion


class Director:
    def __init__(self, transcript: List[Dict]):
        self.transcript = transcript
        self.llm_invoker = LiteLLMInvoker(
            model=config.LLM_SPEC, **config.LLM_PARAMETERS
        )
        self.final_script: List[Dict] = []
        self.tags_injected = 0
        self.max_tags_allowed = len(transcript) * config.MAX_TAG_RATE
        self.global_summary = self._generate_global_summary()

    def _generate_global_summary(self) -> str:
        """
        Generates a high-level summary of the transcript.
        """
        transcript_text = "\n".join([line["text"] for line in self.transcript])
        prompt = config.director_agent["global_summary_prompt"].format(
            transcript_text=transcript_text
        )

        messages = [{"role": "user", "content": prompt}]
        response = self.llm_invoker.invoke(messages)
        return response.content

    def run_rehearsal(self) -> List[Dict]:
        """
        Main method to execute the director-actor workflow.
        """
        for i, line in enumerate(self.transcript):
            if "[insert-verbal-tag-for-pause]" not in line["text"]:
                self.final_script.append(line)
                continue

            local_context = "\n".join(
                [
                    line["text"]
                    for line in self.transcript[
                        max(0, i - 3) : min(len(self.transcript), i + 4)
                    ]
                ]
            )

            unified_prompt = config.director_agent[
                "unified_moment_analysis_prompt"
            ].format(local_context=local_context, current_line=line["text"])
            messages = [{"role": "user", "content": unified_prompt}]
            response_content = self.llm_invoker.invoke(messages).content

            try:
                analysis = json.loads(response_content)
                moment_summary = analysis.get("moment_summary", "")
                directors_note = analysis.get("directors_note", "")
            except json.JSONDecodeError:
                moment_summary = ""
                directors_note = ""

            if self.tags_injected >= self.max_tags_allowed:
                directors_note += " " + config.director_agent["quota_exceeded_note"]

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

            suggestion = get_actor_suggestion(briefing_packet, self.llm_invoker)

            # Check if a new tag was added (not just replacing the placeholder)
            original_has_tag = bool(re.search(r"\([^)]+\)", line["text"]))
            suggestion_has_new_tag = bool(re.search(r"\([^)]+\)", suggestion))
            new_tag_added = suggestion_has_new_tag and not original_has_tag

            if new_tag_added:
                # A new tag was added, check quota
                if self.tags_injected < self.max_tags_allowed:
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
                    self.final_script.append(
                        {
                            "speaker": line["speaker"],
                            "text": suggestion,
                        }
                    )
            else:
                # No new tag was added, just append the suggestion
                self.final_script.append(
                    {
                        "speaker": line["speaker"],
                        "text": suggestion,
                    }
                )

        return self.final_script
