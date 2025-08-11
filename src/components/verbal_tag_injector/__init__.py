"""Verbal tag injector component."""

from .llm_based import build_llm_injector
from .rule_based import rule_based_injector

__all__ = ["build_llm_injector", "rule_based_injector"]
