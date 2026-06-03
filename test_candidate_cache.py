import pytest
from unittest.mock import patch, MagicMock, ANY

from shop_profile_utils import _make_cache_key
from agent import invalidate_candidate_cache, node_retriever


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def make_intent(meal_slots, excluded_tags=None, category_tags=None,
                dietary_hints=None):
    return {
        "city": "京都",
        "region": "jp",
        "meal_slots": meal_slots,
        "excluded_tags": excluded_tags or [],
        "category_tags": category_tags or [],
        "dietary_hints": dietary_hints,
        "time_window": {"start": None, "end": None},
        "mode": "balanced",
        "explicit_constraints": [],
        "wants_flight": False,
        "confidence": 0.9,
        "is_revision": False,
        "is_actionable": True,
        "actionability_followup": None,
        "revision_op": None,
        "confirm_op": None,
        "metadata": {},
        "must_include_shops": [],
        "must_exclude_shops": [],
    }


def make_state_with_intent(intent, meal_type="lunch"):
    from agent import make_initial_state
    state = make_initial_state("test")
    state["intent"] = intent
    state["runtime_services"] = {"llm_router": None}
    state["meal_type"] = meal_type
    return state


# ---------------------------------------------------------------------------
# Tests  –  _make_cache_key
# ---------------------------------------------------------------------------

class TestMakeCacheKey:
    def test_same_params_produce_same_key(self):
        intent = make_intent(["lunch"], excluded_tags=["ramen"])
        k1 = _make_cache_key("lunch", intent)
        k2 = _make_cache_key("lunch", intent)
        assert k1 == k2

    def test_different_meal_type_produce_different_key(self):
        intent = make_intent(["lunch", "dinner"])
        k1 = _make_cache_key("lunch", intent)
        k2 = _make_cache_key("dinner", intent)
        assert k1 != k2

    def test_different_excluded_tags_produce_different_key(self):
        intent1 = make_intent(["lunch"], excluded_tags=["ramen"])
        intent2 = make_intent(["lunch"], excluded_tags=["sushi"])
        k1 = _make_cache_key("lunch", intent1)
        k2 = _make_cache_key("lunch", intent2)
        assert k1 != k2


# ---------------------------------------------------------------------------
# Tests  –  invalidate_candidate_cache
# ---------------------------------------------------------------------------

class TestInvalidateCandidateCache:
    def test_invalidate_removes_entry(self):
        from agent import _candidate_cache
        _candidate_cache["test_key"] = [{"shop": "A"}]
        invalidate_candidate_cache("lunch", make_intent(["lunch"]))
        assert "test_key" not in _candidate_cache

    def test_invalidate_does_not_raise_on_missing(self):
        # Should not raise when key absent
        invalidate_candidate_cache("lunch", make_intent(["lunch"]))


# ---------------------------------------------------------------------------
# Tests  –  node_retriever cache hit / miss
# ---------------------------------------------------------------------------

class TestNodeRetrieverCacheHitMiss:
    @patch("agent.RetrieverAgent")
    @patch("agent._candidate_cache", new_callable=dict)
    def test_cache_hit_does_not_call_retriever(
        self, mock_cache, mock_retriever_cls
    ):
        mock_retriever = MagicMock()
        mock_retriever.run.return_value = MagicMock(as_dict=lambda: {})
        mock_retriever_cls.return_value = mock_retriever

        intent = make_intent(["lunch"])
        state = make_state_with_intent(intent)
        cache_key = _make_cache_key("lunch", intent)

        # Pre-populate cache with something
        mock_cache[cache_key] = {"shop": "cached"}

        with patch.dict("agent._candidate_cache", mock_cache, clear=True):
            result = node_retriever(state)

        assert mock_retriever.run.called is False
        assert result["retrieved"] == {"shop": "cached"}

    @patch("agent.RetrieverAgent")
    @patch("agent._candidate_cache", new_callable=dict)
    def test_cache_miss_calls_retriever_and_populates_cache(
        self, mock_cache, mock_retriever_cls
    ):
        mock_retriever = MagicMock()
        returned = {"shop": "B"}
        mock_retriever.run.return_value = MagicMock(as_dict=lambda: returned)
        mock_retriever_cls.return_value = mock_retriever

        intent = make_intent(["dinner"])
        state = make_state_with_intent(intent)
        cache_key = _make_cache_key("dinner", intent)

        # Cache is empty for this key
        with patch.dict("agent._candidate_cache", mock_cache, clear=True):
            result = node_retriever(state)

        assert mock_retriever.run.called is True, "Retriever should be called on miss"
        assert result["retrieved"] == {"shop": "B"}
        assert mock_cache.get(cache_key) == {"shop": "B"}
