import json
import math
import re
import time
from copy import deepcopy
from functools import wraps
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    TypeAlias,
)

from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.pregel import Pregel
from shared.config import config
from shared.logging import get_logger
from src.components.verbal_tag_injector.procedural_review import procedural_final_cut
from src.components.verbal_tag_injector.state import RehearsalStateModel
from src.components.verbal_tag_injector.token_bucket import TokenBucket

# Import specific config values to avoid scoping issues
DIRECTOR_AGENT_CONFIG = config.director_agent


if TYPE_CHECKING:
    from src.components.verbal_tag_injector.director import Director
else:
    # Runtime placeholder for forward reference
    Director = "Director"

# Initialize logger
logger = get_logger(__name__)


def serialize_token_bucket(bucket: "TokenBucket") -> Dict[str, Any]:
    """Convert TokenBucket to serializable dict."""
    return bucket.to_dict()


def deserialize_token_bucket(data: Dict[str, Any]) -> "TokenBucket":
    """Create TokenBucket from serialized dict."""
    return TokenBucket.from_dict(data)


NodeFunc: TypeAlias = Callable[[RehearsalStateModel], Dict[str, Any]]
NodeDecorator: TypeAlias = Callable[[NodeFunc], NodeFunc]


def compute_global_tags_used(state: RehearsalStateModel) -> int:
    """
    Compute global tag usage from state for resumable budget tracking.

    Args:
        state: The rehearsal state containing original and finalized lines

    Returns:
        Number of tags added across all processed lines
    """
    used = 0
    for orig, fin in zip(state.original_lines, state.finalized_lines, strict=False):
        original_count = len(re.findall(r"\(.*?\)", orig["text"]))
        final_count = len(re.findall(r"\(.*?\)", fin["text"]))
        used += max(0, final_count - original_count)
    return used


def log_node(name: str) -> NodeDecorator:
    """
    Decorator to log node entry/exit and duration.

    Args:
        name: The name of the node for logging

    Returns:
        Decorator function
    """

    def deco(fn: Callable[[RehearsalStateModel], Dict[str, Any]]) -> NodeFunc:
        @wraps(fn)
        def wrapper(state: RehearsalStateModel) -> Dict[str, Any]:
            t0 = time.time()
            current_line = state.current_line_index
            logger.debug(f"Entering node: {name} (line={current_line})")
            result = fn(state)
            dt = time.time() - t0
            keys = list(result.keys()) if isinstance(result, dict) else []
            logger.debug(f"Exiting node: {name} -> keys: {keys} in {dt:.2f}s")
            return result

        return wrapper

    return deco


def log_checkpoint_event(state: RehearsalStateModel, thread_id: str) -> None:
    """
    Log checkpointing events for state persistence.

    Args:
        state: Current rehearsal state
        thread_id: Thread ID for the current execution
    """
    current_line_index = state.current_line_index
    logger.debug(
        f"State checkpointed at line {current_line_index} (thread_id: {thread_id})"
    )


def build_rehearsal_graph(
    director: "Director", checkpointer: BaseCheckpointSaver | bool | None = None
) -> Pregel:
    """
    Build and compile the rehearsal graph with LangGraph.

    Args:
        director: The Director instance for dependencies and methods
        checkpointer: Optional checkpointer for state persistence.
                     If None, uses SqliteSaver with default database.

    Returns:
        Compiled LangGraph with SqliteSaver checkpointing
    """
    # Node function implementations - closures over director for dependencies

    @log_node("initialize")
    def initialize_rehearsal(state: RehearsalStateModel) -> Dict[str, Any]:
        """Generate global summary and initialize rehearsal."""
        # If global_summary already exists, use it (for test compatibility)
        if state.global_summary:
            logger.info(f"Using existing global summary: {state.global_summary}")
            return {}

        transcript_text = "\n".join(
            [f"[{line['speaker']}] {line['text']}" for line in state.original_lines]
        )

        prompt = DIRECTOR_AGENT_CONFIG["global_summary_prompt"].format(
            transcript_text=transcript_text
        )

        messages = [{"role": "user", "content": prompt}]
        response = director.llm_invoker.invoke(messages)
        summary = response.content

        logger.info(f"Global Summary Generated: {summary}")
        return {"global_summary": summary}

    @log_node("define_moment")
    def define_moment(state: RehearsalStateModel) -> Dict[str, Any]:
        """Define moments containing the current line if needed."""
        line_index = state.current_line_index

        # If line already mapped to moment, nothing to do
        if line_index in state.line_to_moment_map:
            return {}

        # Get the last finalized moment for context (currently not used)
        # last_finalized_moment_id = state.moment_cache.get(
        #     "__last_finalized_moment_id__"
        # )
        # Could be used to reconstruct prompt context in future iteration

        # Use director's method to define moments, adapted for state-based context
        moments = director._define_moments_containing(line_index)

        # Update state with new moments
        updated_cache = state.moment_cache.copy()
        updated_map = deepcopy(state.line_to_moment_map)

        for moment in moments:
            moment_id = moment["moment_id"]

            # Validate boundaries (fallback to single-line if invalid)
            if moment["end_line"] < moment["start_line"]:
                logger.warning(
                    f"Invalid moment boundaries for {moment_id}. "
                    f"Creating fallback single-line moment."
                )
                moment = director._create_fallback_moment(line_index)
                moment_id = moment["moment_id"]

            # Store moment in cache
            updated_cache[moment_id] = moment

            # Map all lines in moment to moment_id
            for line_num in range(moment["start_line"], moment["end_line"] + 1):
                if line_num not in updated_map:
                    updated_map[line_num] = []
                updated_map[line_num].append(moment_id)

        # Log the defined moments
        for moment in moments:
            logger.info(
                f"Director defined Moment {moment['moment_id']} "
                f"(lines {moment['start_line']}-{moment['end_line']}) "
                f"because: '{moment.get('description', 'No description')}'"
            )

        return {"moment_cache": updated_cache, "line_to_moment_map": updated_map}

    @log_node("actor_perform_moment")
    def actor_perform_moment(state: RehearsalStateModel) -> Dict[str, Any]:
        """Have Actor perform any moments ending at current line."""
        line_index = state.current_line_index

        # Find moments ending at current line that aren't finalized
        moments_ending_here = [
            moment
            for moment in state.moment_cache.values()
            if (
                isinstance(moment, dict)
                and moment.get("end_line") == line_index
                and not moment.get("is_finalized", False)
            )
        ]

        if not moments_ending_here:
            return {"actor_take": None}

        # Sort by start_line and select primary moment
        moments_ending_here.sort(key=lambda m: m.get("start_line", -1))
        primary = moments_ending_here[0]

        # Build actor script from finalized lines using the moment's line range
        actor_script = []
        for line_num in range(primary["start_line"], primary["end_line"] + 1):
            if 0 <= line_num < len(state.finalized_lines):
                actor_script.append(state.finalized_lines[line_num])

        # Build constraints (pivot line handling)
        constraints = {}
        if primary["lines"]:
            first_line_num = primary["lines"][0]["global_line_number"]
            if (
                state.finalized_lines[first_line_num]
                != state.original_lines[first_line_num]
            ):
                constraints[first_line_num] = (
                    "This line is locked and its content cannot be changed "
                    "as it was the end of the previous moment. Perform around it."
                )
                logger.debug(
                    f"Forward-cascading constraint applied to pivot line "
                    f"{first_line_num} in moment {primary['moment_id']}."
                )

        # Calculate token budget - deserialize TokenBucket if needed
        token_bucket = (
            deserialize_token_bucket(state.token_bucket)
            if isinstance(state.token_bucket, dict)
            else state.token_bucket
        )
        token_bucket.refill(primary["end_line"])
        available_tokens = token_bucket.get_available_tokens()
        max_tags_per_moment = DIRECTOR_AGENT_CONFIG["rate_control"][
            "tag_burst_allowance"
        ]
        moment_token_budget = min(available_tokens, max_tags_per_moment)

        # Check global budget
        from shared.config import config

        new_tag_budget = math.floor(len(state.original_lines) * config.MAX_TAG_RATE)
        global_tags_used = compute_global_tags_used(state)

        logger.debug(
            f"Moment {primary['moment_id']}: Token budget for Actor is "
            f"{moment_token_budget:.1f} tokens."
        )
        logger.debug(f"Global tag budget status: {global_tags_used}/{new_tag_budget}.")

        if global_tags_used >= new_tag_budget:
            logger.debug(
                f"Moment {primary['moment_id']}: Skipping due to exhausted "
                f"global tag budget ({global_tags_used}/{new_tag_budget})."
            )
            return {
                "actor_take": {
                    "moment_id": primary["moment_id"],
                    "skip_due_to_budget": True,
                    "start_line": primary["start_line"],
                    "end_line": primary["end_line"],
                }
            }

        # Call Actor to perform the moment
        logger.info(f"--- Processing Moment {primary['moment_id']} ---")
        logger.info(
            f"Moment description: {primary.get('description', 'No description')}"
        )
        logger.info(
            f"Moment director's notes: {primary.get('directors_notes', 'No notes')}"
        )

        actor_result = director.actor.perform_moment(
            moment_id=primary["moment_id"],
            lines=actor_script,
            token_budget=moment_token_budget,
            constraints=constraints,
            global_summary=state.global_summary,
        )

        return {
            "actor_take": {
                "moment_id": primary["moment_id"],
                "result": actor_result,
                "start_line": primary["start_line"],
                "end_line": primary["end_line"],
            }
        }

    def compute_tag_budget_for_review(state: RehearsalStateModel) -> int:
        """Compute strict per-moment tag budget for Director's review."""
        # Calculate global budget constraints
        from shared.config import config as shared_config

        global_budget_total = math.floor(
            len(state.original_lines) * shared_config.MAX_TAG_RATE
        )
        global_used_so_far = compute_global_tags_used(state)
        remaining_global_budget = max(0, global_budget_total - global_used_so_far)

        # Get moment bucket availability
        token_bucket = (
            deserialize_token_bucket(state.token_bucket)
            if isinstance(state.token_bucket, dict)
            else state.token_bucket
        )
        line_index = state.current_line_index
        token_bucket.refill(line_index)
        moment_bucket_available = token_bucket.get_available_tokens()

        # Apply constraints
        max_tags_per_moment = shared_config.director_agent["rate_control"][
            "tag_burst_allowance"
        ]
        tag_budget_for_review = min(
            remaining_global_budget, moment_bucket_available, max_tags_per_moment
        )

        return int(tag_budget_for_review)

    @log_node("procedural_review")
    def procedural_review_node(state: RehearsalStateModel) -> Dict[str, Any]:
        """Apply procedural review to the Actor's performance."""
        actor_take = state.actor_take
        if not actor_take or actor_take.get("skip_due_to_budget"):
            return {"reviewed_take": None, "review_mode_used": "procedural"}

        moment_id = actor_take["moment_id"]
        start_line = actor_take["start_line"]
        end_line = actor_take["end_line"]

        # Build dictionaries for procedural review
        original_by_line = {}
        performed_by_line = {}

        for line_num in range(start_line, end_line + 1):
            if 0 <= line_num < len(state.original_lines):
                original_by_line[line_num] = state.original_lines[line_num]["text"]
                if line_num in actor_take["result"]:
                    performed_by_line[line_num] = actor_take["result"][line_num]["text"]
                else:
                    performed_by_line[line_num] = state.original_lines[line_num]["text"]

        # Compute tag budget
        tag_budget_for_review = compute_tag_budget_for_review(state)

        # Count actor's added tags for logging
        actor_tags_added = 0
        for line_num in range(start_line, end_line + 1):
            if line_num in original_by_line and line_num in performed_by_line:
                orig_tags = len(re.findall(r"\(.*?\)", original_by_line[line_num]))
                perf_tags = len(re.findall(r"\(.*?\)", performed_by_line[line_num]))
                actor_tags_added += max(0, perf_tags - orig_tags)

        budget_remaining = tag_budget_for_review
        overage = max(0, actor_tags_added - tag_budget_for_review)

        logger.info(
            f"Moment {moment_id} budget analysis: {actor_tags_added} added, "
            f"{budget_remaining} remaining, {overage} over budget"
        )

        # Apply procedural final cut
        final_by_line, metrics = procedural_final_cut(
            original_by_line, performed_by_line, tag_budget_for_review
        )

        # Count final tags for compliance check
        final_tag_count = 0
        for line_num in range(start_line, end_line + 1):
            if line_num in final_by_line:
                final_tag_count += len(re.findall(r"\(.*?\)", final_by_line[line_num]))

        # Count original tags for compliance comparison
        original_tag_count = 0
        for line_num in range(start_line, end_line + 1):
            if line_num in original_by_line:
                original_tag_count += len(
                    re.findall(r"\(.*?\)", original_by_line[line_num])
                )

        is_compliant = (final_tag_count - original_tag_count) <= tag_budget_for_review

        logger.info(
            f"Director's Final Cut (Procedural): Removed {metrics['removed']} tags "
            f"from Moment {moment_id} to meet budget"
        )
        logger.debug(
            f"Final moment quality: {final_tag_count} tags, "
            f"budget compliance: {is_compliant}"
        )

        return {
            "reviewed_take": final_by_line,
            "review_mode_used": "procedural",
            "review_metrics": metrics,
        }

    @log_node("llm_review")
    def llm_review_node(state: RehearsalStateModel) -> Dict[str, Any]:
        """Apply LLM-based review to the Actor's performance."""
        review_start_time = time.time()
        actor_take = state.actor_take

        if not actor_take or actor_take.get("skip_due_to_budget"):
            return {
                "reviewed_take": None,
                "review_mode_used": "llm",
                "llm_review_failed": False,
            }

        moment_id = actor_take["moment_id"]
        start_line = actor_take["start_line"]
        end_line = actor_take["end_line"]

        try:
            # Build context for LLM prompt
            tag_budget_for_review = compute_tag_budget_for_review(state)

            # Original script text for this moment
            original_script_lines = []
            for line_num in range(start_line, end_line + 1):
                if 0 <= line_num < len(state.original_lines):
                    line = state.original_lines[line_num]
                    original_script_lines.append(f"[{line['speaker']}] {line['text']}")
            original_script_text = "\n".join(original_script_lines)

            # Actor's performance text for this moment
            actor_performance_lines = []
            for line_num in range(start_line, end_line + 1):
                if line_num in actor_take["result"]:
                    line = actor_take["result"][line_num]
                    actor_performance_lines.append(
                        f"[{line['speaker']}] {line['text']}"
                    )
                elif 0 <= line_num < len(state.original_lines):
                    line = state.original_lines[line_num]
                    actor_performance_lines.append(
                        f"[{line['speaker']}] {line['text']}"
                    )
            actor_performance_text = "\n".join(actor_performance_lines)

            # Previous moment context (simplified for now)
            last_moment_summary = "No previous moment"
            previous_moment_performance_text = "None"
            if state.last_finalized_moment_id:
                last_moment_summary = f"Moment {state.last_finalized_moment_id}"
                previous_moment_performance_text = (
                    "[Previous moment content not available in this context]"
                )

            # Format the director review prompt
            prompt = DIRECTOR_AGENT_CONFIG["director_review_prompt"].format(
                global_summary=state.global_summary,
                last_moment_summary=last_moment_summary,
                previous_moment_performance_text=previous_moment_performance_text,
                original_script_text=original_script_text,
                actor_performance_text=actor_performance_text,
                tag_budget=tag_budget_for_review,
                start_line=start_line,
                start_line_plus_1=start_line + 1,
            )

            # Invoke LLM
            messages = [{"role": "user", "content": prompt}]
            response = director.llm_invoker.invoke(messages)

            # Parse JSON response
            cleaned_response = response.content.strip()
            match = re.search(r"\{.*\}", cleaned_response, re.DOTALL)
            if not match:
                raise ValueError("No JSON object found in LLM response")

            json_text = match.group(0)
            llm_decision = json.loads(json_text)

            # Validate and build reviewed_take
            reviewed_take = {}
            total_actor_tags = 0
            kept_tags = 0

            for line_num in range(start_line, end_line + 1):
                line_key = f"line_{line_num}"
                if line_key in llm_decision:
                    reviewed_take[line_num] = llm_decision[line_key]

                    # Count tags for metrics
                    if line_num in actor_take["result"]:
                        original_text = state.original_lines[line_num]["text"]
                        actor_text = actor_take["result"][line_num]["text"]
                        final_text = llm_decision[line_key]

                        orig_tags = len(re.findall(r"\(.*?\)", original_text))
                        actor_tags = len(re.findall(r"\(.*?\)", actor_text))
                        final_tags = len(re.findall(r"\(.*?\)", final_text))

                        total_actor_tags += max(0, actor_tags - orig_tags)
                        kept_tags += max(0, final_tags - orig_tags)
                elif 0 <= line_num < len(state.original_lines):
                    # Fall back to original text
                    reviewed_take[line_num] = state.original_lines[line_num]["text"]

            review_duration = time.time() - review_start_time
            logger.info(
                f"Director's Final Cut (LLM): Kept {kept_tags}/{total_actor_tags} "
                f"Actor tags in Moment {moment_id}"
            )
            logger.debug(
                f"Review process for Moment {moment_id} completed in "
                f"{review_duration:.2f}s"
            )

            return {
                "reviewed_take": reviewed_take,
                "review_mode_used": "llm",
                "llm_review_failed": False,
                "review_metrics": {
                    "kept_tags": kept_tags,
                    "total_actor_tags": total_actor_tags,
                    "review_duration": review_duration,
                },
            }

        except Exception as e:
            review_duration = time.time() - review_start_time
            error_msg = f"LLM review error: {str(e)}"
            logger.warning(
                f"LLM review failed for Moment {moment_id}, falling back to "
                f"procedural review"
            )
            logger.debug(f"Error details: {error_msg}")

            return {
                "llm_review_failed": True,
                "review_metrics": {
                    "error": error_msg,
                    "review_duration": review_duration,
                },
            }

    # Router functions for Director's Final Cut
    def route_to_review_mode(state: RehearsalStateModel) -> str:
        """Route to appropriate review mode based on configuration and actor results."""
        actor_take = state.actor_take

        # Skip review if no actor take or budget was already exceeded
        if not actor_take or actor_take.get("skip_due_to_budget"):
            logger.debug(
                "Review router directing to: finalize_moment "
                "(skip_review - no actor take or budget exceeded)"
            )
            return "finalize_moment"

        # Get the configured review mode
        from shared.config import config as shared_config

        review_mode = shared_config.director_agent["review"]["mode"]
        moment_id = actor_take.get("moment_id", "unknown")

        logger.info(
            f"Director's Final Cut: Using '{review_mode}' review mode for "
            f"Moment {moment_id}"
        )

        next_node = "llm_review" if review_mode == "llm" else "procedural_review"

        logger.debug(f"Review router directing to: {next_node} (mode: {review_mode})")
        return next_node

    def route_post_llm_review(state: RehearsalStateModel) -> str:
        """Route after LLM review - fallback to procedural or finalize."""
        if state.llm_review_failed:
            logger.debug(
                "Post-LLM review router directing to: procedural_review (fallback)"
            )
            return "procedural_review"
        else:
            logger.debug("Post-LLM review router directing to: finalize_moment")
            return "finalize_moment"

    @log_node("finalize_moment")
    def finalize_moment(state: RehearsalStateModel) -> Dict[str, Any]:
        """Finalize moments, update state, and advance line index."""
        line_index = state.current_line_index

        # Find moments ending at current line
        moments_ending_here = [
            moment
            for moment in state.moment_cache.values()
            if (
                isinstance(moment, dict)
                and moment.get("end_line") == line_index
                and not moment.get("is_finalized", False)
            )
        ]

        if not moments_ending_here:
            # Log checkpoint event for state persistence
            # Note: This is where the state will be checkpointed by LangGraph
            # We're simulating the logging that would happen during actual checkpointing
            logger.debug(f"State checkpointed at line {line_index}")
            return {"current_line_index": line_index + 1, "actor_take": None}

        # Sort by start_line
        moments_ending_here.sort(key=lambda m: m.get("start_line", -1))
        primary = moments_ending_here[0]

        updated_cache = state.moment_cache.copy()
        updated_lines = state.finalized_lines.copy()
        # Deserialize TokenBucket if needed
        updated_bucket = (
            deserialize_token_bucket(state.token_bucket)
            if isinstance(state.token_bucket, dict)
            else state.token_bucket
        )

        # Handle actor take results
        actor_take = state.actor_take
        if (
            not actor_take
            or actor_take.get("moment_id") != primary["moment_id"]
            or actor_take.get("skip_due_to_budget")
        ):
            # Mark primary as finalized without processing
            updated_cache[primary["moment_id"]]["is_finalized"] = True

            # Mark co-terminous moments as finalized
            for other_moment in moments_ending_here[1:]:
                moment_id = other_moment["moment_id"]
                logger.debug(
                    f"Moment {moment_id} is co-terminous with "
                    f"'{primary['moment_id']}'. Marking as finalized without action."
                )
                updated_cache[moment_id]["is_finalized"] = True

            # Update last finalized moment
            last_finalized_moment_id = primary["moment_id"]

            if actor_take and actor_take.get("skip_due_to_budget"):
                logger.info(
                    f"--- Moment {primary['moment_id']} finalized "
                    "(skipped due to budget) ---"
                )
            else:
                logger.info(
                    f"--- Moment {primary['moment_id']} finalized "
                    "(no processing needed) ---"
                )

        else:
            # Process actor results through Director's Final Cut
            start_time = time.time()

            # Use reviewed_take if available (from Director's Final Cut),
            # otherwise fall back to actor result
            if state.reviewed_take:
                # Apply Director's Final Cut decisions
                for line_num, final_text in state.reviewed_take.items():
                    if 0 <= line_num < len(updated_lines):
                        # Preserve original structure but update text
                        updated_line = updated_lines[line_num].copy()
                        updated_line["text"] = final_text
                        updated_lines[line_num] = updated_line

                # Calculate tags spent based on final reviewed version
                director_result = {}
                for line_num, final_text in state.reviewed_take.items():
                    if 0 <= line_num < len(state.original_lines):
                        director_result[line_num] = {
                            "speaker": state.original_lines[line_num]["speaker"],
                            "text": final_text,
                            "global_line_number": line_num,
                        }
            else:
                # Fall back to original actor result (backward compatibility)
                director_result = actor_take["result"]

                # Update finalized lines with actor's original result
                for line_number, line_obj in director_result.items():
                    updated_lines[line_number] = line_obj

            # Calculate tags spent and update budget
            tags_spent = director._calculate_tags_spent(
                primary["lines"], director_result
            )
            logger.info(f"Moment {primary['moment_id']} spent {tags_spent:.2f} tokens")

            updated_bucket.spend(tags_spent)

            # Mark primary as finalized
            updated_cache[primary["moment_id"]]["is_finalized"] = True

            # Mark co-terminous moments as finalized
            for other_moment in moments_ending_here[1:]:
                moment_id = other_moment["moment_id"]
                logger.debug(
                    f"Moment {moment_id} is co-terminous with "
                    f"'{primary['moment_id']}'. Marking as finalized without action."
                )
                updated_cache[moment_id]["is_finalized"] = True

            # Update last finalized moment
            last_finalized_moment_id = primary["moment_id"]

            # Log completion
            duration = time.time() - start_time
            logger.info(f"Moment {primary['moment_id']} finalized in {duration:.2f}s.")
            logger.info(f"--- Moment {primary['moment_id']} finalized ---")

        result = {
            "finalized_lines": updated_lines,
            "moment_cache": updated_cache,
            "current_line_index": line_index + 1,
            "token_bucket": serialize_token_bucket(updated_bucket),  # Serialize
            "actor_take": None,
            "last_finalized_moment_id": last_finalized_moment_id,
            # Reset review fields to avoid cross-moment bleed
            "reviewed_take": None,
            "review_mode_used": None,
            "llm_review_failed": False,
            "review_metrics": None,
        }

        # Log checkpoint event for state persistence
        # Note: This is where the state will be checkpointed by LangGraph
        # We're simulating the logging that would happen during actual checkpointing
        logger.debug(f"State checkpointed at line {line_index}")

        return result

    # Router function
    def should_continue_rehearsal(state: RehearsalStateModel) -> str:
        """Route to continue processing or end."""
        # Check if all lines have been processed
        if state.current_line_index >= len(state.original_lines):
            logger.debug(
                "Router 'should_continue_rehearsal' routing to: END "
                "(all lines processed)"
            )
            return END

        logger.debug("Router 'should_continue_rehearsal' routing to: define_moment")
        return "define_moment"

    builder: StateGraph[RehearsalStateModel] = StateGraph(RehearsalStateModel)

    # Add nodes with explicit types
    builder.add_node(
        "initialize",
        RunnableLambda(initialize_rehearsal),
        input_schema=RehearsalStateModel,
    )
    builder.add_node(
        "define_moment", RunnableLambda(define_moment), input_schema=RehearsalStateModel
    )
    builder.add_node(
        "actor_perform_moment",
        RunnableLambda(actor_perform_moment),
        input_schema=RehearsalStateModel,
    )
    # Director's Final Cut nodes
    builder.add_node(
        "procedural_review",
        RunnableLambda(procedural_review_node),
        input_schema=RehearsalStateModel,
    )
    builder.add_node(
        "llm_review",
        RunnableLambda(llm_review_node),
        input_schema=RehearsalStateModel,
    )
    builder.add_node(
        "finalize_moment",
        RunnableLambda(finalize_moment),
        input_schema=RehearsalStateModel,
    )

    # Set up edges
    builder.set_entry_point("initialize")

    # From initialize, go to conditional routing
    builder.add_conditional_edges("initialize", should_continue_rehearsal)

    # Main loop edges
    builder.add_edge("define_moment", "actor_perform_moment")

    # Director's Final Cut routing: actor_perform_moment -> route_to_review_mode
    builder.add_conditional_edges(
        "actor_perform_moment",
        route_to_review_mode,
        {
            "finalize_moment": "finalize_moment",
            "procedural_review": "procedural_review",
            "llm_review": "llm_review",
        },
    )

    # Procedural review always goes to finalize
    builder.add_edge("procedural_review", "finalize_moment")

    # LLM review routing: llm_review -> route_post_llm_review
    builder.add_conditional_edges(
        "llm_review",
        route_post_llm_review,
        {
            "finalize_moment": "finalize_moment",
            "procedural_review": "procedural_review",
        },
    )

    # From finalize_moment back to conditional routing
    builder.add_conditional_edges("finalize_moment", should_continue_rehearsal)
    # Compile with checkpointer
    if checkpointer is None:
        # Use SqliteSaver (persistent checkpointing) if no checkpointer provided
        # Create SqliteSaver with a direct connection
        import sqlite3

        from shared.config import config

        conn = sqlite3.connect(config.REHEARSAL_CHECKPOINT_PATH)
        checkpointer = SqliteSaver(conn)

    graph = builder.compile(checkpointer=checkpointer)

    logger.info(
        f"LangGraph compiled with {len(builder.nodes)} nodes and "
        f"SqliteSaver checkpointing (persistent)"
    )

    return graph
