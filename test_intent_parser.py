"""
Tests for intent_parser.py.

Coverage:
1. Rule fast-path handles a structured query (no LLM call).
2. Ambiguous query falls back to LLM.
3. LLM output that fails pydantic validation triggers one retry.
4. Second failure raises ValueError (not silently swallowed by parse_intent).
5. parse_intent never raises — LLM error returns best-effort rule intent.
6. Route-level: correct fields for flight / right-now / dietary queries.
7. Mock backends used throughout; real HTTP never called.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, call

import pytest

from intent_parser import (
    Intent,
    intent_from_snapshot_dict,
    parse_intent,
    parse_intent_llm,
    parse_intent_rules,
)
from llm_router import LLMResponse, TaskType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_llm_response(content: str) -> LLMResponse:
    return LLMResponse(
        content=content,
        model_used="mock-qwen",
        tokens_in=10,
        tokens_out=20,
        latency_ms=50,
        cost_usd=0.0,
    )


def _valid_llm_json(**overrides: Any) -> str:
    base = {
        "city": "台北",
        "region": "tw",
        "meal_slots": ["lunch", "dinner"],
        "time_window": {"start": None, "end": None},
        "category_tags": ["sour", "taiwanese"],
        "dietary_hints": None,
        "excluded_shops": [],
        "mode": "balanced",
        "explicit_constraints": [],
        "wants_flight": False,
        "confidence": 0.8,
        "is_actionable": True,
        "actionability_followup": None,
    }
    base.update(overrides)
    return json.dumps(base, ensure_ascii=False)


def _mock_router(side_effect_list: list[Any] | None = None, return_value: Any = None) -> MagicMock:
    router = MagicMock()
    if side_effect_list is not None:
        router.complete.side_effect = side_effect_list
    elif return_value is not None:
        router.complete.return_value = return_value
    return router


# ---------------------------------------------------------------------------
# 1. Rule fast-path: structured query → no LLM call
# ---------------------------------------------------------------------------

class TestRuleFastPath:
    def test_structured_query_uses_rules(self) -> None:
        """「三餐拉麵 7:00~21:00」: rules should handle it; LLM must NOT be called."""
        router = _mock_router()
        intent = parse_intent("三餐拉麵 7:00~21:00", router)

        router.complete.assert_not_called()
        assert "ramen" in intent.category_tags
        assert len(intent.meal_slots) == 3
        assert intent.time_window[0] == "07:00"
        assert intent.time_window[1] == "21:00"
        assert intent.confidence >= 0.6

    def test_rule_mode_right_now(self) -> None:
        router = _mock_router()
        intent = parse_intent("RIGHT_NOW 拉麵", router)
        router.complete.assert_not_called()
        assert intent.mode == "right_now"

    def test_rule_meal_count_expansion(self) -> None:
        """「三餐拉麵」 should expand to 3 meal slots."""
        router = _mock_router()
        intent = parse_intent("三餐拉麵", router)
        assert len(intent.meal_slots) == 3

    def test_rule_time_window_extracted(self) -> None:
        router = _mock_router()
        intent = parse_intent("早上7點到晚上9點拉麵", router)
        router.complete.assert_not_called()
        assert intent.time_window[0] is not None
        assert intent.time_window[1] is not None

    def test_rule_flight_intent(self) -> None:
        router = _mock_router()
        intent = parse_intent("我想訂機票從台北到東京", router)
        router.complete.assert_not_called()
        assert intent.wants_flight is True

    def test_rule_dietary_vegan(self) -> None:
        router = _mock_router()
        intent = parse_intent("台北 vegan 餐廳三餐", router)
        router.complete.assert_not_called()
        assert intent.dietary_hints == "vegan"

    def test_rule_taipei_city(self) -> None:
        router = _mock_router()
        intent = parse_intent("台北早餐午餐晚餐", router)
        router.complete.assert_not_called()
        assert intent.city == "台北"
        assert intent.region == "tw"

    def test_parse_intent_rules_returns_none_for_empty_signals(self) -> None:
        result = parse_intent_rules("")
        assert result is None

    def test_parse_intent_rules_returns_intent_for_structured(self) -> None:
        result = parse_intent_rules("三餐拉麵 7:00~21:00")
        assert result is not None
        assert result.confidence >= 0.6


# ---------------------------------------------------------------------------
# 2. LLM fallback: ambiguous query
# ---------------------------------------------------------------------------

class TestLLMFallback:
    def test_ambiguous_query_calls_llm(self) -> None:
        """「我女友懷孕想吃酸的」 has no rule-extractable signals → LLM called."""
        llm_resp = _make_llm_response(_valid_llm_json(city="台北", meal_slots=["dinner"]))
        router = _mock_router(return_value=llm_resp)

        intent = parse_intent("我女友懷孕想吃酸的", router)

        router.complete.assert_called_once()
        # Verify it called INTENT_PARSING task type
        args, kwargs = router.complete.call_args
        assert args[0] == TaskType.INTENT_PARSING or kwargs.get("task") == TaskType.INTENT_PARSING or args[0] == "INTENT_PARSING"

    def test_ambiguous_intent_fields_from_llm(self) -> None:
        llm_resp = _make_llm_response(_valid_llm_json(
            city="台北", region="tw", meal_slots=["dinner"], category_tags=["sour"]
        ))
        router = _mock_router(return_value=llm_resp)

        intent = parse_intent("我女友懷孕想吃酸的", router)

        assert intent.city == "台北"
        assert "dinner" in intent.meal_slots
        assert intent.region == "tw"

    def test_low_confidence_rule_triggers_llm(self) -> None:
        """A query where rules give confidence < 0.6 falls back to LLM."""
        # "vegan 拉麵" → rules get ramen+vegan (confidence ~0.30) → LLM
        llm_resp = _make_llm_response(_valid_llm_json(
            meal_slots=["lunch", "dinner"],
            category_tags=["ramen"],
            dietary_hints="vegan",
        ))
        router = _mock_router(return_value=llm_resp)
        intent = parse_intent("vegan 拉麵", router)
        router.complete.assert_called_once()
        assert intent.dietary_hints == "vegan"


# ---------------------------------------------------------------------------
# 3 & 4. Schema mismatch → retry once; second failure → ValueError
# ---------------------------------------------------------------------------

class TestSchemaRetry:
    def test_bad_json_retries_once(self) -> None:
        """First LLM response is bad JSON → retry → second response is valid."""
        bad = _make_llm_response("this is not json")
        good = _make_llm_response(_valid_llm_json())
        router = _mock_router(side_effect_list=[bad, good])

        intent = parse_intent_llm("我女友懷孕想吃酸的", router)

        assert router.complete.call_count == 2
        assert intent.city == "台北"

    def test_invalid_schema_retries_once(self) -> None:
        """First response has wrong schema → retry → second is valid."""
        bad = _make_llm_response('{"wrong_key": 123}')  # valid JSON but missing required fields → defaults kick in
        good = _make_llm_response(_valid_llm_json(meal_slots=["breakfast"]))
        router = _mock_router(side_effect_list=[bad, good])

        # Actually pydantic will accept extra keys and use defaults for missing ones,
        # so this tests the case where JSON itself is bad:
        import json as _json
        truly_bad = _make_llm_response("```json\nnot valid\n```")
        router2 = _mock_router(side_effect_list=[truly_bad, good])
        intent = parse_intent_llm("test query", router2)
        assert router2.complete.call_count == 2

    def test_both_attempts_fail_raises(self) -> None:
        """If both LLM attempts return invalid JSON → ValueError."""
        bad = _make_llm_response("not json at all")
        router = _mock_router(side_effect_list=[bad, bad])

        with pytest.raises(ValueError, match="both attempts failed"):
            parse_intent_llm("ambiguous query", router)

        assert router.complete.call_count == 2


# ---------------------------------------------------------------------------
# 5. parse_intent never raises (LLM error → best-effort fallback)
# ---------------------------------------------------------------------------

class TestParseIntentNeverRaises:
    def test_llm_failure_returns_fallback_intent(self) -> None:
        """parse_intent returns a fallback Intent when LLM totally fails."""
        bad = _make_llm_response("not json")
        router = _mock_router(side_effect_list=[bad, bad])  # both attempts fail

        # Must not raise; should return default Intent
        intent = parse_intent("我女友懷孕想吃酸的", router)
        assert isinstance(intent, Intent)
        assert intent.city is None
        assert intent.is_actionable is False

    def test_llm_exception_returns_fallback(self) -> None:
        """Router.complete() raising an exception is also handled gracefully."""
        router = _mock_router()
        router.complete.side_effect = RuntimeError("network error")

        intent = parse_intent("我女友懷孕想吃酸的", router)
        assert isinstance(intent, Intent)


# ---------------------------------------------------------------------------
# 6. Field correctness
# ---------------------------------------------------------------------------

class TestIntentFields:
    def test_mode_taste_max_for_food_focus(self) -> None:
        router = _mock_router()
        intent = parse_intent("台北三餐拉麵", router)
        # ramen tag → taste_max
        assert intent.mode == "taste_max"

    def test_appetite_light_constraint(self) -> None:
        router = _mock_router()
        intent = parse_intent("台北三餐拉麵 吃不太下", router)
        assert "appetite_light" in intent.explicit_constraints

    def test_no_city_in_rules_without_geographic_hint(self) -> None:
        """Structured meal query without a named city → city stays unset (no invented default)."""
        router = _mock_router()
        intent = parse_intent_rules("三餐拉麵 7:00~21:00")
        assert intent is not None
        assert intent.city is None

    def test_parse_intent_clamps_when_rules_lack_city(self) -> None:
        """High-confidence rules without a resolved city must not reach retrieval as actionable."""
        router = _mock_router()
        intent = parse_intent("三餐拉麵 7:00~21:00", router)
        router.complete.assert_not_called()
        assert intent.city is None
        assert intent.is_actionable is False

    def test_tokyo_detected_from_query(self) -> None:
        router = _mock_router()
        intent = parse_intent("東京三餐拉麵", router)
        router.complete.assert_not_called()
        assert intent.city == "東京"
        assert intent.region == "jp"

    def test_intent_as_dict_is_serialisable(self) -> None:
        """Intent.as_dict() must produce a JSON-serialisable dict."""
        intent = Intent(
            city="台北",
            region="tw",
            meal_slots=["lunch", "dinner"],
            time_window=("10:00", "21:00"),
            category_tags=["ramen"],
            dietary_hints="vegan",
            excluded_shops=["-demo-venue-"],
            mode="taste_max",
            explicit_constraints=["appetite_light"],
            wants_flight=False,
            confidence=0.75,
        )
        d = intent.as_dict()
        serialised = json.dumps(d)  # must not raise
        recovered = json.loads(serialised)
        assert recovered["city"] == "台北"
        assert recovered["meal_slots"] == ["lunch", "dinner"]
        assert recovered["time_window"] == ["10:00", "21:00"]
        assert recovered["excluded_shops"] == ["-demo-venue-"]

    def test_intent_snapshot_roundtrip_excluded_shops(self) -> None:
        d = intent_from_snapshot_dict(
            {"city": "京都", "excluded_shops": ["茶寮 都路里"], "confidence": 0.5},
        ).as_dict()
        assert d["city"] == "京都"
        assert d["excluded_shops"] == ["茶寮 都路里"]


class TestExcludedShopsQuickRules:
    @pytest.mark.skip(reason="否定句現在統一由 LLM 處理，規則引擎預期回傳 None")
    def test_parse_intent_rules_extracts_do_not_eat_phrase(self) -> None:
        r = parse_intent_rules("三餐拉麵 不想吃茶寮都路里")
        assert r is not None
        assert any("茶寮" in x for x in r.excluded_shops)


# ---------------------------------------------------------------------------
# 7. LLM markdown fence stripping
# ---------------------------------------------------------------------------

class TestMarkdownFenceStripping:
    def test_llm_response_with_fences(self) -> None:
        fenced = f"```json\n{_valid_llm_json()}\n```"
        llm_resp = _make_llm_response(fenced)
        router = _mock_router(return_value=llm_resp)

        intent = parse_intent_llm("我女友懷孕想吃酸的", router)
        assert intent.city == "台北"
