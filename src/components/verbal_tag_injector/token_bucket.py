"""Token bucket implementation for rate limiting tag injection."""

from typing import Any, Dict


class TokenBucket:
    """A token bucket for rate limiting tag injection."""

    def __init__(self, rate: float, burst_allowance: float):
        self.rate = rate  # tokens per line
        self.burst_allowance = burst_allowance  # maximum tokens
        self.tokens = burst_allowance  # current tokens
        self.last_line_index = -1  # last line processed

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "rate": self.rate,
            "burst_allowance": self.burst_allowance,
            "tokens": self.tokens,
            "last_line_index": self.last_line_index,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TokenBucket":
        """Create from dictionary after deserialization."""
        instance = cls(data["rate"], data["burst_allowance"])
        instance.tokens = data["tokens"]
        instance.last_line_index = data["last_line_index"]
        return instance

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
