from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class RehearsalStateModel(BaseModel):
    original_lines: List[Dict[str, Any]]
    finalized_lines: List[Dict[str, Any]]
    moment_cache: Dict[str, Dict[str, Any]]
    line_to_moment_map: Dict[int, List[str]]
    global_summary: str
    token_bucket: Any
    current_line_index: int
    actor_take: Optional[Dict[str, Any]]
    last_finalized_moment_id: Optional[str] = None
    # Director's Final Cut review fields
    reviewed_take: Optional[Dict[int, str]] = (
        None  # mapping of global line number to final text
    )
    review_mode_used: Optional[str] = None
    llm_review_failed: Optional[bool] = False
    review_metrics: Optional[Dict[str, Any]] = None
