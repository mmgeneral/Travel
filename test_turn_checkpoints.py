"""Tests for per-conversation-turn checkpoint bookkeeping (LangGraph undo / history)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graph_checkpoint_utils import _graph_resolve_checkpoint_snapshot, extend_turn_checkpoint_in_state
from orchestrator import _merge_continuation_invoke_state


def test_extend_turn_checkpoint_dedupes_and_append() -> None:
    s = {"turn_checkpoints": ["a"], "final_itinerary": "x"}
    extend_turn_checkpoint_in_state(s, {"configurable": {"checkpoint_id": "a"}})
    assert [e["id"] for e in s["turn_checkpoints"]] == ["a"]
    extend_turn_checkpoint_in_state(s, {"configurable": {"checkpoint_id": "b"}})
    assert [e["id"] for e in s["turn_checkpoints"]] == ["a", "b"]


def test_extend_turn_checkpoint_missing_config_noop() -> None:
    s: dict[str, object] = {"turn_checkpoints": ["z"]}
    extend_turn_checkpoint_in_state(s, None)
    extend_turn_checkpoint_in_state(s, {})
    assert s["turn_checkpoints"] == ["z"]


def test_merge_continuation_preserves_turn_checkpoints() -> None:
    merged = _merge_continuation_invoke_state(
        {
            "turn_checkpoints": ["a1f2-b3", "c4d5-e6"],
            "research_log": [],
            "final_itinerary": "東京",
            "dynamic_shop_pool": [],
            "dietary_profile": {"ethics": "unspecified"},
            "intent": {},
        },
        query="下一轮",
        agent_run_id="thread-x",
        checkpoint_thread_id="thread-x",
        dietary_profile=None,
        advanced_mode=False,
        user_locale=None,
        user_lat=None,
        user_lng=None,
    )
    assert [e["id"] for e in merged["turn_checkpoints"]] == ["a1f2-b3", "c4d5-e6"]


@pytest.mark.asyncio
async def test_resolve_cp_index_uses_turn_checkpoints_list() -> None:
    snap = SimpleNamespace(values={"turn_checkpoints": ["uu1", "uu2"]})
    s2 = SimpleNamespace(
        values={"final_itinerary": "itin2"},
        config={"configurable": {"checkpoint_id": "uu2"}},
    )
    hist_match = [
        SimpleNamespace(config={"configurable": {"checkpoint_id": "uu0"}}),
        SimpleNamespace(config={"configurable": {"checkpoint_id": "uu1"}}),
        s2,
    ]
    graph = MagicMock()

    async def fake_get_state(_graph: MagicMock, _cfg: dict) -> SimpleNamespace:
        return snap

    with (
        patch("graph_checkpoint_utils._graph_get_state", new=AsyncMock(side_effect=fake_get_state)),
        patch(
            "graph_checkpoint_utils._graph_history_chronological",
            new=AsyncMock(return_value=hist_match),
        ),
    ):
        resolved = await _graph_resolve_checkpoint_snapshot(graph, "tid", "cp_002")
        assert resolved is s2
        resolved_bad = await _graph_resolve_checkpoint_snapshot(graph, "tid", "cp_099")
        assert resolved_bad is None


@pytest.mark.asyncio
async def test_resolve_cp_fallback_full_history_when_turn_list_missing() -> None:
    hist_fb = [
        SimpleNamespace(config={"configurable": {"checkpoint_id": "x1"}}),
        SimpleNamespace(config={"configurable": {"checkpoint_id": "x2"}}),
    ]
    graph = MagicMock()
    snap_empty = SimpleNamespace(values={})
    with (
        patch(
            "graph_checkpoint_utils._graph_get_state",
            new=AsyncMock(return_value=snap_empty),
        ),
        patch(
            "graph_checkpoint_utils._graph_history_chronological",
            new=AsyncMock(return_value=hist_fb),
        ),
    ):
        out = await _graph_resolve_checkpoint_snapshot(graph, "tid", "cp_002")
        assert out is hist_fb[1]
