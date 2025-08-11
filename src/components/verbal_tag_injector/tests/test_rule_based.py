"""Unit tests for the rule-based verbal tag injector."""

import random

from shared.config import config
from src.components.verbal_tag_injector.rule_based import rule_based_injector


def test_rule_based_injector_with_placeholder(monkeypatch):
    """Test rule-based injector with pause placeholder."""
    state = {
        "prev_lines": [],
        "current_line": f"[S1] Hello there {config.PAUSE_PLACEHOLDER} how are you?",
        "next_lines": [],
        "summary": "Greetings",
        "topic": "introduction",
    }

    # Mock random.choice to return a specific value for testing
    monkeypatch.setattr(random, "choice", lambda x: x[0])

    result = rule_based_injector(state)
    # Should replace the placeholder
    assert config.PAUSE_PLACEHOLDER not in result["modified_line"]
    assert result["modified_line"] != state["current_line"]
    # Should use the first line combiner
    assert config.LINE_COMBINERS[0] in result["modified_line"]


def test_rule_based_injector_without_placeholder(monkeypatch):
    """Test rule-based injector without pause placeholder."""
    state = {
        "prev_lines": [],
        "current_line": "[S1] Hello there",
        "next_lines": ["[S2] Hi, how are you?"],
        "summary": "Greetings",
        "topic": "introduction",
    }

    # Mock random.random to always be less than MAX_TAG_RATE
    monkeypatch.setattr(random, "random", lambda: 0.01)

    result = rule_based_injector(state)
    # Should add a verbal tag at the start
    assert any(tag in result["modified_line"] for tag in config.VERBAL_TAGS)
