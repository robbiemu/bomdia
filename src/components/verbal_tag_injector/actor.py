import json
import re
from typing import Dict, List

from shared.config import config
from shared.llm_invoker import LiteLLMInvoker
from shared.logging import get_logger

# Initialize logger
logger = get_logger(__name__)


class Actor:
    def __init__(self, llm_invoker: LiteLLMInvoker) -> None:
        self.llm_invoker = llm_invoker

    def perform_moment(
        self,
        moment_id: str,
        lines: List[Dict],
        token_budget: float,
        constraints: Dict,
        global_summary: str,
        sample_name: str,
    ) -> Dict:
        """
        Perform a moment by processing all lines in the moment with holistic context.

        Args:
            moment_id: Identifier for the moment
            lines: List of line dictionaries to process
            token_budget: Available token budget for this moment
            constraints: Dictionary of constraints for specific lines
            global_summary: Global summary of the entire transcript
            sample_name: A name from the script

        Returns:
            Dict mapping line numbers to processed line objects
        """
        logger.info(f"[Actor] Performing moment {moment_id} with {len(lines)} lines")

        # Create a context string with all lines in the moment, including line numbers
        moment_lines_text = []
        for line in lines:
            line_num = line["global_line_number"]
            speaker = line.get("speaker_name") or line["speaker"]
            moment_lines_text.append(f"line_{line_num}: [{speaker}] {line['text']}")
        moment_text = "\n".join(moment_lines_text)

        # Build constraints message if any
        constraints_text = ""
        if constraints:
            constraints_list = []
            for line_num, constraint in constraints.items():
                constraints_list.append(f"Line {line_num}: {constraint}")
            constraints_text = "\nCONSTRAINTS:\n" + "\n".join(constraints_list)

        # Format the prompt with all context
        prompt = self._format_prompt(
            moment_text=moment_text,
            global_summary=global_summary,
            token_budget=token_budget,
            constraints_text=constraints_text,
            start_line=lines[0]["global_line_number"],
            sample_name=sample_name,
        )

        logger.debug(f"Final prompt for Actor:\n{prompt}")

        system_prompt = config.actor_agent.get("system_prompt", "")
        messages = (
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            if system_prompt
            else [{"role": "user", "content": prompt}]
        )
        logger.debug("Sending prompt to LLM for moment performance...")
        response = self.llm_invoker.invoke(messages)
        logger.debug(f"Raw LLM response for moment performance:\n{response.content}")

        # Process the response to extract tagged lines
        tagged_lines = self._parse_response(response.content, lines)

        # Return a dictionary mapping line numbers to tagged line objects
        result = {}
        for i, line in enumerate(lines):
            line_number = line["global_line_number"]
            if i < len(tagged_lines):
                # Create a new line object with the tagged text
                tagged_line = line.copy()
                tagged_line["text"] = tagged_lines[i]
                result[line_number] = tagged_line
            else:
                # Fallback to original line if parsing failed
                result[line_number] = line

        return result

    def _format_prompt(
        self,
        moment_text: str,
        global_summary: str,
        token_budget: float,
        constraints_text: str,
        start_line: int,
        sample_name: str,
    ) -> str:
        """
        Format the prompt for the LLM based on the task directive template.
        """

        # Prepare the lists of available tags by formatting them from the config
        verbal_tags_list = ", ".join([f"`{tag}`" for tag in config.VERBAL_TAGS])
        line_combiners_list = ", ".join([f"`{tag}`" for tag in config.LINE_COMBINERS])

        prompt_template = config.actor_agent["moment_task_directive_template"]
        prompt = prompt_template.format(
            moment_text=moment_text,
            global_summary=global_summary,
            token_budget=token_budget,
            constraints_text=constraints_text,
            available_verbal_tags=verbal_tags_list,
            available_line_combiners=line_combiners_list,
            start_line=start_line,
            start_line_plus_1=start_line + 1,
            sample_name=sample_name,
        )
        return str(prompt)  # Explicitly cast to str to satisfy MyPy

    def _parse_response(
        self, response_text: str, original_lines: List[Dict]
    ) -> List[str]:
        """
        Parse the LLM response to extract tagged lines.

        Args:
            response_text: The raw response from the LLM
            original_lines: The original lines that were sent to the LLM

        Returns:
            List of tagged line texts in the same order as original_lines
        """
        try:
            # Try to parse as JSON
            response_data = json.loads(response_text)

            # Extract lines in order
            tagged_lines = []
            # Loop through the original lines to find the expected GLOBAL keys
            for line in original_lines:
                global_line_num = line["global_line_number"]
                line_key = f"line_{global_line_num}"

                if line_key in response_data:
                    raw_text = response_data[line_key]
                    # Defensively strip any leading speaker tags the LLM might add.
                    cleaned_text = re.sub(r"^\s*\[.*?\]\s*", "", raw_text)
                    tagged_lines.append(cleaned_text)
                else:
                    logger.warning(
                        f"Key '{line_key}' not found in Actor's JSON response. "
                        f"Falling back to original text for line {global_line_num}."
                    )
                    # Fallback to original text if not found
                    tagged_lines.append(line["text"])
            return tagged_lines

        except json.JSONDecodeError:
            # Fallback to text parsing if JSON fails
            logger.warning(
                f"Actor response parsing failed, falling back to text parsing: "
                f"{response_text[:100]}..."
            )

            # Split the response into lines
            response_lines = response_text.strip().split("\n")

            # Clean up the lines (remove empty lines and speaker tags if present)
            tagged_lines = []
            for response_line in response_lines:
                stripped_line = response_line.strip()
                if stripped_line:
                    # Remove speaker tags if present
                    if stripped_line.startswith("[") and "]" in stripped_line:
                        # Find closing bracket and remove the speaker tag
                        close_bracket = stripped_line.find("]")
                        if close_bracket > 0:
                            stripped_line = stripped_line[close_bracket + 1 :].strip()
                    tagged_lines.append(stripped_line)

            # If we didn't get the right number of lines, return original lines
            if len(tagged_lines) != len(original_lines):
                logger.warning(
                    "Actor response parsing mismatch: expected "
                    f"{len(original_lines)} lines, "
                    f"got {len(tagged_lines)}. Returning original lines."
                )
                return [line["text"] for line in original_lines]

            return tagged_lines
