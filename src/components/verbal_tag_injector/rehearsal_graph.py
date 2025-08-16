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

        prompt = config.director_agent["global_summary_prompt"].format(
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

    @log_node("pass_through_review")
    def pass_through_review(state: RehearsalStateModel) -> Dict[str, Any]:
        """Placeholder for Director's Final Cut step."""
        return {}

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
            # Process actor results
            start_time = time.time()
            director_result = actor_take["result"]  # Pass through for now

            # Calculate tags spent and update budget
            tags_spent = director._calculate_tags_spent(
                primary["lines"], director_result
            )
            logger.info(f"Moment {primary['moment_id']} spent {tags_spent:.2f} tokens")

            updated_bucket.spend(tags_spent)

            # Update finalized lines
            for line_number, line_obj in director_result.items():
                updated_lines[line_number] = line_obj

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
    builder.add_node(
        "pass_through_review",
        RunnableLambda(pass_through_review),
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
    builder.add_edge("actor_perform_moment", "pass_through_review")
    builder.add_edge("pass_through_review", "finalize_moment")

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
