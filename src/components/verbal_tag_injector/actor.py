from shared.llm_invoker import LiteLLMInvoker


def get_actor_suggestion(briefing_packet: dict, llm_invoker: LiteLLMInvoker) -> str:
    """
    Constructs a prompt for the LLM based on the briefing_packet and returns the
    suggestion.
    """
    prompt = briefing_packet["task_directive_template"].format(
        global_summary=briefing_packet["global_summary"],
        local_context=briefing_packet["local_context"],
        moment_summary=briefing_packet["moment_summary"],
        directors_notes=briefing_packet["directors_notes"],
        current_line=briefing_packet["current_line"],
    )

    messages = [{"role": "user", "content": prompt}]
    response = llm_invoker.invoke(messages)
    return response.content
