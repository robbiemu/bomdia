import json
import math
import re
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional

from shared.config import config
from shared.llm_invoker import LiteLLMInvoker
from shared.logging import get_logger
from src.components.verbal_tag_injector.actor import Actor

# Initialize logger
logger = get_logger(__name__)


class TokenBucket:
    """A token bucket for rate limiting tag injection."""

    def __init__(self, rate: float, burst_allowance: float):
        self.rate = rate  # tokens per line
        self.burst_allowance = burst_allowance  # maximum tokens
        self.tokens = burst_allowance  # current tokens
        self.last_line_index = -1  # last line processed

    def refill(self, current_line_index: int) -> None:
        """Refill the token bucket based on lines processed."""
        if self.last_line_index >= 0:
            lines_processed = current_line_index - self.last_line_index
            new_tokens = lines_processed * self.rate
            self.tokens = min(self.burst_allowance, self.tokens + new_tokens)
        self.last_line_index = current_line_index

    def get_available_tokens(self) -> float:
        """Get the current number of available tokens."""
        return self.tokens

    def spend(self, tokens: float) -> None:
        """Spend tokens from the bucket."""
        self.tokens = max(0.0, self.tokens - tokens)


class Director:
    def __init__(self, transcript: List[Dict]):
        self.transcript = transcript
        self.llm_invoker = LiteLLMInvoker(
            model=config.LLM_SPEC, **config.LLM_PARAMETERS
        )
        # Initialize state management - work directly with flat list of lines
        self.original_lines = self._add_global_line_numbers(transcript)
        self.finalized_lines = deepcopy(self.original_lines)
        self.moment_cache: Dict[str, Dict] = {}
        self.line_to_moment_map: Dict[int, List[str]] = {}
        self.actor = Actor(self.llm_invoker)

        # Calculate the total budget of NEW tags we can inject (deterministic approach)
        self.new_tag_budget = math.floor(len(self.original_lines) * config.MAX_TAG_RATE)
        logger.info(
            f"Director initialized with a budget of {self.new_tag_budget} "
            f"new verbal tags."
        )
        self.global_summary = self._generate_global_summary()

        # Initialize the central token bucket for global pacing
        self.token_bucket = TokenBucket(
            rate=config.director_agent["rate_control"]["target_tag_rate"],
            burst_allowance=config.director_agent["rate_control"][
                "tag_burst_allowance"
            ],
        )

        # Track finalized moments
        self.finalized_moments: set[str] = set()

        # Track the last finalized moment for context
        self.last_finalized_moment: Optional[Dict[str, Any]] = None

        # Track global tag usage
        self.global_tags_used: float = 0.0

    def _add_global_line_numbers(self, lines: List[Dict]) -> List[Dict]:
        """
        Add global line numbers to line objects.
        """
        numbered_lines = []
        for i, line in enumerate(lines):
            new_line = line.copy()
            new_line["global_line_number"] = i
            numbered_lines.append(new_line)
        return numbered_lines

    def _generate_global_summary(self) -> str:
        """
        Generates a high-level summary of the transcript.
        """
        # The generation itself is a background detail. The result is what's important.
        logger.debug("Generating global summary...")
        transcript_text = "\n".join(
            [f"[{line['speaker']}] {line['text']}" for line in self.original_lines]
        )
        prompt = config.director_agent["global_summary_prompt"].format(
            transcript_text=transcript_text
        )

        messages = [{"role": "user", "content": prompt}]
        response = self.llm_invoker.invoke(messages)
        # The summary itself is key context, so log it at INFO.
        logger.info(f"Global Summary Generated: {response.content}")
        return response.content

    def _is_candidate_for_tagging(self, line: str) -> bool:
        """
        Triage function to determine if a line is a candidate for tagging.
        """
        # Check for keywords indicating emotion or action
        emotional_keywords = ["cries", "shouts", "love", "suddenly"]
        for keyword in emotional_keywords:
            if keyword in line.lower():
                return True

        # Check for punctuation indicating strong emotion (! or ?)
        if line.endswith("!") or line.endswith("?"):
            return True

        # Check for parenthetical actions
        return bool(re.search(r"\(.*?\)", line))

    def _define_moments_containing(self, line_number: int) -> List[Dict[str, Any]]:
        """
        Define narrative moments containing the given line number.
        Uses LLM to analyze the transcript and identify natural narrative boundaries.
        """
        # Get the last finalized moment for context
        last_moment_summary = "None yet"
        last_moment_end_line = -1
        last_finalized_line_text = "None"

        if self.last_finalized_moment:
            last_moment_summary = self.last_finalized_moment.get(
                "description", "No description"
            )
            last_moment_end_line = self.last_finalized_moment.get("end_line", -1)
            if last_moment_end_line >= 0 and last_moment_end_line < len(
                self.finalized_lines
            ):
                last_finalized_line_text = self.finalized_lines[last_moment_end_line][
                    "text"
                ]

        # Get context around this line (a few lines before and after)
        start_line = max(0, line_number - 3)
        end_line = min(len(self.original_lines), line_number + 4)

        # Create script slice with line numbers
        forward_script_slice = []
        for i in range(start_line, end_line):
            line_obj = self.original_lines[i]
            forward_script_slice.append(
                f"{i}: [{line_obj['speaker']}] {line_obj['text']}"
            )

        forward_script_slice_text = "\n".join(forward_script_slice)

        # Create prompt for LLM to define moments using the creative-first approach
        # Use initial prompt when no previous moment exists, otherwise use the
        #  standard prompt
        previous_moment_text = ""
        if self.last_finalized_moment:
            previous_moment_text = config.director_agent[
                "previous_moment_template"
            ].format(
                last_moment_summary=last_moment_summary,
                last_moment_end_line=last_moment_end_line,
                last_finalized_line_text=last_finalized_line_text,
            )
        prompt = config.director_agent["moment_definition_prompt"].format(
            previous_moment_segment=previous_moment_text,
            forward_script_slice_text=forward_script_slice_text,
            line_number=line_number,
        )

        messages = [{"role": "user", "content": prompt}]
        try:
            response = self.llm_invoker.invoke(messages)

            # Try to parse JSON response
            try:
                # Clean up the response content by removing markdown code blocks
                # . and other text
                cleaned_response = response.content.strip()
                # Find the JSON object using a regular expression
                match = re.search(r"\{.*\}", cleaned_response, re.DOTALL)
                if match:
                    json_text = match.group(0)
                    moment_direction = json.loads(json_text)
                else:
                    # Fallback if no JSON object is found
                    raise json.JSONDecodeError(
                        "No JSON object found in response", cleaned_response, 0
                    )

                # Convert to our internal moment format
                moments = []

                # Create the moment with validated boundaries
                moment_lines = []
                start_line = moment_direction.get("start_line", line_number)
                end_line = moment_direction.get("end_line", line_number)

                # Validate moment boundaries
                if start_line > end_line:
                    logger.warning(
                        f"Director returned invalid moment boundaries: "
                        f"start_line ({start_line}) > end_line ({end_line}). "
                        f"Using fallback single-line moment."
                    )
                    start_line = line_number
                    end_line = line_number

                # Ensure boundaries are within valid range
                start_line = max(0, min(start_line, len(self.original_lines) - 1))
                end_line = max(0, min(end_line, len(self.original_lines) - 1))

                for ln in range(start_line, end_line + 1):
                    if 0 <= ln < len(self.original_lines):
                        moment_lines.append(self.original_lines[ln])

                moment = {
                    "moment_id": f"moment_{start_line}_{end_line}",
                    "start_line": start_line,
                    "end_line": end_line,
                    "is_finalized": False,
                    "lines": moment_lines,
                    "description": moment_direction.get("moment_summary", ""),
                    "directors_notes": moment_direction.get("directors_notes", ""),
                }
                moments.append(moment)

                return moments

            except json.JSONDecodeError:
                logger.warning(
                    f"Failed to parse JSON response for moment definition: "
                    f"{response.content[:100]}..."
                )
                # Fallback to single-line moment
                return [self._create_fallback_moment(line_number)]

        except Exception as e:
            logger.warning(f"Error defining moments containing line {line_number}: {e}")
            # Fallback to single-line moment
            return [self._create_fallback_moment(line_number)]

    def _create_fallback_moment(self, line_number: int) -> Dict[str, Any]:
        """
        Create a fallback single-line moment.
        """
        moment = {
            "moment_id": f"fallback_moment_{line_number}",
            "start_line": line_number,
            "end_line": line_number,
            "is_finalized": False,
            "lines": [self.original_lines[line_number]],
            "description": "Fallback single-line moment",
            "directors_notes": "Process as a single line moment",
        }

        return moment

    def _find_moments_ending_at(self, line_number: int) -> List[Dict]:
        """
        Find all moments that end at the given line number.
        """
        moments = []
        for _moment_id, moment in self.moment_cache.items():
            if moment.get("end_line") == line_number and not moment.get(
                "is_finalized", False
            ):
                moments.append(moment)

        # Error out if more than 2 moments end at the same line
        if len(moments) > 2:
            raise RuntimeError(
                f"More than 2 moments ending at line {line_number}: "
                f"{len(moments)} moments found"
            )

        return moments

    def _execute_full_moment(self, moment: Dict) -> None:
        """
        Execute the full Actor/Director workflow for a moment.
        """
        moment_id = moment["moment_id"]
        logger.info(f"--- Processing Moment {moment_id} ---")
        logger.info(
            f"Moment description: {moment.get('description', 'No description')}"
        )
        logger.info(
            f"Moment director's notes: {moment.get('directors_notes', 'No notes')}"
        )

        # Record start time for performance metrics
        start_time = time.time()

        # Prepare Actor's Data
        # Gather all line_objs for the moment from the finalized_lines structure
        actor_script = [
            self.finalized_lines[line["global_line_number"]] for line in moment["lines"]
        ]

        constraints = {}

        # Check only the first line for pivot constraints (already edited in
        #  previous moment)
        if moment["lines"]:  # Safety check for empty moments
            first_line = moment["lines"][0]
            first_line_number = first_line["global_line_number"]
            current_first_line = self.finalized_lines[first_line_number]

            # Check if this is a pivot line (already edited)
            if current_first_line != self.original_lines[first_line_number]:
                # This line was already edited, so it's locked
                constraints[first_line_number] = (
                    "This line is locked and its content cannot be changed "
                    "as it was the end of the previous moment. Perform around it."
                )
                logger.debug(
                    f"Forward-cascading constraint applied to pivot line "
                    f"{first_line_number} in moment {moment_id}."
                )

        # Refill the token bucket based on progress
        # Use the end line of the moment as the current line index for refilling
        current_line_index = moment["end_line"]
        self.token_bucket.refill(current_line_index)

        # Calculate token budget for this moment from the central token bucket
        available_tokens = self.token_bucket.get_available_tokens()
        # Cap the budget for a single moment to prevent moment-greedy behavior
        max_tags_per_moment = config.director_agent["rate_control"][
            "tag_burst_allowance"
        ]
        moment_token_budget = min(available_tokens, max_tags_per_moment)

        logger.debug(
            f"Moment {moment_id}: Token budget for Actor is "
            f"{moment_token_budget:.1f} tokens."
        )
        logger.debug(
            f"Global tag budget status: {self.global_tags_used}/{self.new_tag_budget}."
        )

        # Check if we have enough budget to process this moment
        if self.global_tags_used >= self.new_tag_budget:
            logger.debug(
                f"Moment {moment_id}: Skipping due to exhausted global tag budget "
                f"({self.global_tags_used}/{self.new_tag_budget})."
            )
            # Mark as finalized without processing
            moment["is_finalized"] = True
            self.finalized_moments.add(moment_id)
            logger.info(f"--- Moment {moment_id} finalized (skipped due to budget) ---")
            return

        # Actor's Performance (One Call)
        logger.debug("Delegating to Actor for creative suggestion...")
        actor_result = self.actor.perform_moment(
            moment_id=moment_id,
            lines=actor_script,
            token_budget=moment_token_budget,
            constraints=constraints,
            global_summary=self.global_summary,
        )

        # Director's Review (One Call)
        logger.debug("Delegating to Director for final review...")
        director_result = self._review_and_finalize_moment(
            moment_id=moment_id,
            actor_takes=actor_result,
            constraints=constraints,
            original_lines=moment["lines"],
            directors_notes=moment.get("directors_notes", ""),
        )

        # Calculate how many tags were actually added/spent
        tags_spent = self._calculate_tags_spent(moment["lines"], director_result)
        logger.info(f"Moment {moment_id} spent {tags_spent:.2f} tokens")

        # Update global tag usage
        self.global_tags_used += tags_spent

        # Spend the tokens from the central token bucket
        self.token_bucket.spend(tags_spent)

        # Recomposition
        for line_number, final_line_obj in director_result.items():
            # Update the finalized_lines structure directly
            self.finalized_lines[line_number] = final_line_obj

        # Record end time and log performance
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"Moment {moment_id} finalized in {duration:.2f}s.")

        # Update State
        # Mark the moment as finalized
        moment["is_finalized"] = True
        self.finalized_moments.add(moment_id)
        self.last_finalized_moment = moment

        logger.info(f"--- Moment {moment_id} finalized ---")

    def _calculate_tags_spent(
        self, original_lines: List[Dict], final_lines: Dict
    ) -> float:
        """
        Calculate how many tags were actually added/spent in this moment.
        """
        tags_spent = 0.0

        for line in original_lines:
            line_number = line["global_line_number"]
            if line_number in final_lines:
                original_text = line["text"]
                final_text = final_lines[line_number]["text"]

                # Count tags in original and final text
                original_tag_count = len(re.findall(r"\(.*?\)", original_text))
                final_tag_count = len(re.findall(r"\(.*?\)", final_text))

                # Calculate tags added
                tags_added = max(0, final_tag_count - original_tag_count)
                tags_spent += tags_added

        return tags_spent

    def _review_and_finalize_moment(
        self,
        moment_id: str,
        actor_takes: Dict,
        constraints: Dict,
        original_lines: List[Dict],
        directors_notes: str,
    ) -> Dict:
        """
        Director's review and finalization of the Actor's takes.
        This is where the Director can make final edits to the Actor's performance.
        """
        # For now, we'll implement a simple review that ensures budget compliance
        # In a more sophisticated implementation, this would do more complex curation

        final_takes = {}

        # Copy actor takes to final takes
        for line_number, line_obj in actor_takes.items():
            final_takes[line_number] = line_obj.copy()

        # Simple validation: ensure we don't exceed our tag budget
        # Count existing tags in original lines
        original_tags = 0
        new_tags = 0

        for line in original_lines:
            line_number = line["global_line_number"]
            if line_number in actor_takes:
                original_text = line["text"]
                new_text = actor_takes[line_number]["text"]

                # Count tags in original and new text
                original_tag_count = len(re.findall(r"\(.*?\)", original_text))
                new_tag_count = len(re.findall(r"\(.*?\)", new_text))

                original_tags += original_tag_count
                new_tags += new_tag_count

        # If we've exceeded our tag budget, we might need to strip some tags
        # This is a simplified approach - a more sophisticated implementation
        # would be more selective about which tags to keep/remove
        tags_added = new_tags - original_tags
        if tags_added > 0:
            logger.debug(f"Tags added for moment {moment_id}: {tags_added}")
            # Check if we're exceeding our global budget
            if self.global_tags_used + tags_added > self.new_tag_budget:
                overage = (self.global_tags_used + tags_added) - self.new_tag_budget
                logger.debug(
                    f"Director Review: Actor exceeded budget. "
                    f"Requested {tags_added}, budget was "
                    f"{self.new_tag_budget - self.global_tags_used}. "
                    f"Stripping {overage} tags."
                )
                # In a real implementation, we would do more sophisticated
                # tag management here
                # For now, we'll just pass through the actor's takes

        return final_takes

    def run_rehearsal(self) -> List[Dict]:
        """
        Main method to execute the moment-based director-actor workflow.
        """
        logger.info(
            f"Starting moment-based rehearsal process for "
            f"{len(self.original_lines)} lines..."
        )

        # Main Processing Loop - iterate through each line
        for line_number in range(len(self.original_lines)):
            self.original_lines[line_number]

            # STAGE 1: Discover moment if needed.
            if line_number not in self.line_to_moment_map:
                # Define moment(s), state is updated
                moments = self._define_moments_containing(line_number)

                # Update state for each moment
                for moment in moments:
                    moment_id = moment["moment_id"]

                    # Validate moment boundaries
                    if moment["end_line"] < moment["start_line"]:
                        logger.warning(
                            f"Invalid moment boundaries for moment "
                            f"{moment_id}. Creating fallback single-line moment."
                        )
                        moment = self._create_fallback_moment(line_number)
                        moment_id = moment["moment_id"]

                    # Update state
                    self.moment_cache[moment_id] = moment
                    # Map all lines in the moment to the moment_id
                    for line_num in range(moment["start_line"], moment["end_line"] + 1):
                        if line_num not in self.line_to_moment_map:
                            self.line_to_moment_map[line_num] = []
                        self.line_to_moment_map[line_num].append(moment_id)

                    # Add INFO level logging for successfully defined moments
                    logger.info(
                        f"Director defined Moment {moment_id} "
                        f"(lines {moment['start_line']}-{moment['end_line']}) "
                        f"because: '{moment.get('description', 'No description')}'"
                    )

            # STAGE 2: Check if this line completes a moment.
            moments_ending_here = self._find_moments_ending_at(line_number)

            # Sort moments by their starting line number to ensure
            # chronological processing
            # This ensures that moments are always processed in the
            # order they appear in the script
            moments_ending_here.sort(key=lambda m: m.get("start_line", -1))

            # Optimize finalization for co-terminous moments
            if len(moments_ending_here) > 1:
                logger.debug(
                    f"Found {len(moments_ending_here)} co-terminous moments "
                    f"ending at line {line_number}"
                )

                # Process only the first moment with full workflow
                first_moment = moments_ending_here[0]
                if not first_moment.get("is_finalized", False):
                    logger.debug(
                        f"Processing first moment {first_moment['moment_id']} with "
                        "full workflow"
                    )
                    self._execute_full_moment(first_moment)

                # Mark other moments as finalized without action
                for i in range(1, len(moments_ending_here)):
                    other_moment = moments_ending_here[i]
                    if not other_moment.get("is_finalized", False):
                        other_moment_id = other_moment["moment_id"]
                        logger.debug(
                            f"Moment {other_moment_id} is co-terminous with an "
                            "already-processed moment. Marking as finalized without "
                            "action."
                        )
                        other_moment["is_finalized"] = True
                        self.finalized_moments.add(other_moment_id)
            else:
                # Normal case: process single moment
                for moment in moments_ending_here:
                    if not moment.get("is_finalized", False):
                        # The moment is complete. Execute the full Actor/Director
                        #  workflow.
                        self._execute_full_moment(moment)

        # Return the finalized lines without global_line_number
        final_script = []
        for line in self.finalized_lines:
            # Remove the global_line_number as it was only for internal processing
            line_copy = line.copy()
            if "global_line_number" in line_copy:
                del line_copy["global_line_number"]
            final_script.append(line_copy)

        logger.info("--- Moment-based rehearsal process complete. ---")
        return final_script
