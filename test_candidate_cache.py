import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from shop_profile_utils import _make_cache_key
from agent import invalidate_candidate_cache, make_initial_state


def make_intent(meal_slots, excluded_tags=None, category_tags=None, dietary_hints=None):
    return {
        "city": "京都", "region": "jp",
        "meal_slots": meal_slots,
        "excluded_tags": excluded_tags or [],
        "category_tags": category_tags or [],
        "dietary_hints": dietary_hints,
        "time_window": {"start": None, "end": None},
        "mode": "balanced", "explicit_constraints": [],
        "wants_flight": False, "confidence": 0.9,
        "is_revision": False, "is_actionable": True,
        "actionability_followup": None,
        "revision_op": None, "confirm_op": None, "metadata": {},
        "must_include_shops": [], "must_exclude_shops": [],
    }


# ── _make_cache_key ──────────────────────────────────────

def test_make_cache_key_same_intent_same_key():
    intent = make_intent(["lunch"], excluded_tags=["beef"])
    assert _make_cache_key("lunch", intent) == _make_cache_key("lunch", intent)


def test_make_cache_key_different_tags_different_key():
    a = make_intent(["lunch"], excluded_tags=["beef"])
    b = make_intent(["lunch"], excluded_tags=["sushi"])
    assert _make_cache_key("lunch", a) != _make_cache_key("lunch", b)


def test_make_cache_key_different_meal_type_different_key():
    intent = make_intent(["lunch", "dinner"])
    assert _make_cache_key("lunch", intent) != _make_cache_key("dinner", intent)


def test_make_cache_key_tag_order_invariant():
    a = make_intent(["lunch"], excluded_tags=["beef", "sushi"])
    b = make_intent(["lunch"], excluded_tags=["sushi", "beef"])
    assert _make_cache_key("lunch", a) == _make_cache_key("lunch", b)


# ── invalidate_candidate_cache ───────────────────────────

def test_invalidate_candidate_cache_clears_all():
    state = make_initial_state("test")
    state["candidate_cache"] = {"key1": ["店A"], "key2": ["店B"]}
    result = invalidate_candidate_cache(state)
    assert result["candidate_cache"] == {}


def test_invalidate_candidate_cache_empty_state_no_error():
    state = make_initial_state("test")
    state["candidate_cache"] = {}
    result = invalidate_candidate_cache(state)
    assert result["candidate_cache"] == {}


# ── node_retriever cache hit ─────────────────────────────

@pytest.mark.asyncio
async def test_cache_hit_skips_retriever():
    from agent import node_retriever

    intent = make_intent(["lunch", "dinner"])
    state = make_initial_state("test")
    state["intent"] = intent
    state["runtime_services"] = {"llm_router": MagicMock()}

    # Pre-populate cache for both slots
    cache = {
        _make_cache_key("lunch", intent): ["松籟庵"],
        _make_cache_key("dinner", intent): ["燃えよ麺助"],
    }
    state["candidate_cache"] = cache

    with patch("agent.RetrieverAgent") as MockRetriever:
        result = await node_retriever(state)

    MockRetriever.assert_not_called()
    assert result["researcher_candidate_names"] == ["松籟庵", "燃えよ麺助"]


# ── node_retriever cache miss ────────────────────────────

@pytest.mark.asyncio
async def test_cache_miss_runs_retriever_and_populates_cache():
    from agent import node_retriever

    intent = make_intent(["lunch"])
    state = make_initial_state("test")
    state["intent"] = intent
    state["candidate_cache"] = {}  # empty
    state["runtime_services"] = {"llm_router": MagicMock()}
    state["researcher_candidate_names"] = ["松籟庵"]  # 模擬 retriever 跑完的結果

    mock_report = MagicMock()
    mock_report.candidates = []
    mock_report.city = "京都"
    mock_report.seed_count = 0
    mock_report.dynamic_count = 0
    mock_report.notes = []
    mock_report.gaps = []
    mock_report.as_dict.return_value = {}

    with patch("agent.RetrieverAgent") as MockRetriever:
        MockRetriever.return_value.arun = AsyncMock(return_value=mock_report)
        result = await node_retriever(state)

    MockRetriever.assert_called_once()
    key = _make_cache_key("lunch", intent)
    assert key in result["candidate_cache"]
    assert "松籟庵" in result["candidate_cache"][key]
