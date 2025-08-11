from typing import List, TypedDict


class VerbalTagInjectorState(TypedDict):
    """Represents the state of our verbal tag injection graph."""

    # Input fields for the node
    prev_lines: List[str]
    current_line: str
    next_lines: List[str]
    summary: str
    topic: str
    # Output field from the node
    modified_line: str
