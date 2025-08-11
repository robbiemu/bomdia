from shared.llm_invoker import LiteLLMInvoker
from shared.logging import get_logger

# Initialize logger
logger = get_logger(__name__)


def get_actor_suggestion(briefing_packet: dict, llm_invoker: LiteLLMInvoker) -> str:
    """
    Constructs a prompt for the LLM based on the briefing_packet and returns the
    suggestion.
    """
    # Log the briefing packet details at DEBUG level
    logger.debug(
        f"Actor briefing - Global summary: {briefing_packet['global_summary']}"
    )
    logger.debug(
        f"Actor briefing - Moment summary: {briefing_packet['moment_summary']}"
    )
    logger.debug(
        f"Actor briefing - Director's notes: {briefing_packet['directors_notes']}"
    )
    logger.debug(f"Actor briefing - Current line: {briefing_packet['current_line']}")

    prompt = briefing_packet["task_directive_template"].format(
        global_summary=briefing_packet["global_summary"],
        local_context=briefing_packet["local_context"],
        moment_summary=briefing_packet["moment_summary"],
        directors_notes=briefing_packet["directors_notes"],
        current_line=briefing_packet["current_line"],
    )

    messages = [{"role": "user", "content": prompt}]
    logger.debug("Sending prompt to LLM for actor suggestion...")
    response = llm_invoker.invoke(messages)
    logger.debug("Received actor suggestion from LLM.")
    return response.content
