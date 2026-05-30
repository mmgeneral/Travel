"""Helpers for ``turn_checkpoints`` rewind semantics."""
from __future__ import annotations

from graph_checkpoint_utils import turn_checkpoints_trimmed_to_checkpoint_id


def test_trim_preserves_prefix_through_target() -> None:
    result = turn_checkpoints_trimmed_to_checkpoint_id(["a", "b", "c"], "b")
    assert [e["id"] for e in result] == ["a", "b"]


def test_trim_missing_returns_none() -> None:
    assert turn_checkpoints_trimmed_to_checkpoint_id(["a", "b"], "z") is None
