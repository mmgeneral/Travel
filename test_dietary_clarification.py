"""Unit tests for dietary strict/loose clarification helpers (graph node in agent.py)."""
from __future__ import annotations

from agent import (
    _ambiguous_dietary_hint_for_clarification,
    _parse_dietary_clarification_reply,
)


def test_parse_dietary_clarification_reply_basic() -> None:
    assert _parse_dietary_clarification_reply("A") == "strict"
    assert _parse_dietary_clarification_reply("(B)") == "loose"
    assert _parse_dietary_clarification_reply("  選項A ") == "strict"
    assert _parse_dietary_clarification_reply("hello") is None


def test_ambiguous_hint_single_key_only() -> None:
    resolved: dict[str, str] = {}
    assert (
        _ambiguous_dietary_hint_for_clarification(
            {"dietary_hints": "no_beef", "is_revision": False},
            resolved,
        )
        == "no_beef"
    )
    assert _ambiguous_dietary_hint_for_clarification({"dietary_hints": "vegan"}, resolved) is None
    assert (
        _ambiguous_dietary_hint_for_clarification(
            {"dietary_hints": "no_beef,no_pork"},
            resolved,
        )
        is None
    )
    assert (
        _ambiguous_dietary_hint_for_clarification({"dietary_hints": "no_beef"}, {"no_beef": "loose"})
        is None
    )
