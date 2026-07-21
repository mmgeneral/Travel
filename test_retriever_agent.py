"""
Tests for agents/retriever.py — RetrieverAgent and RetrievalReport.

Coverage
--------
1. LLM is actually called (RETRIEVAL_REASONING) for candidate reasoning.
2. RetrievalReport.gaps correctly flags a missing city catalog
   (e.g. "東京" requested but only Kyoto catalog available in seed).
3. Retriever does NOT return a final itinerary — only candidates + notes.
4. as_dict() is JSON-serialisable.
5. Structural gap detection works for uncovered meal slots and category tags.
6. Empty candidate pool produces an informative gap, not a crash.
7. LLM failure falls back gracefully; notes=[], gaps contains error string.
8. NearbySearchTool failure is absorbed; seed-only result is still valid.
"""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.retriever import (
    RetrievalReport,
    RetrieverAgent,
    _detect_structural_gaps,
    _has_dedicated_catalog,
    _load_seed_for_city,
)
from llm_router import LLMResponse, TaskType
from shop_planning import ShopProfile, BookingType, QueueStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _llm_response(notes: list[str], gaps: list[str]) -> LLMResponse:
    content = json.dumps({"notes": notes, "gaps": gaps}, ensure_ascii=False)
    return LLMResponse(
        content=content,
        model_used="mock-gemini",
        tokens_in=50,
        tokens_out=30,
        latency_ms=80,
        cost_usd=0.0,
    )


def _mock_router(notes: list[str] | None = None, gaps: list[str] | None = None) -> MagicMock:
    router = MagicMock()
    router.complete.return_value = _llm_response(
        notes or ["燃えよ麺助: classic Kyoto ramen, good fit for lunch slot."],
        gaps or [],
    )
    return router


def _minimal_state(
    city: str = "京都",
    region: str = "jp",
    meal_slots: list[str] | None = None,
    category_tags: list[str] | None = None,
    query: str = "京都ランチ",
) -> dict:
    return {
        "query": query,
        "intent": {
            "city": city,
            "region": region,
            "meal_slots": meal_slots or ["lunch"],
            "category_tags": category_tags or [],
            "dietary_hints": None,
            "mode": "balanced",
            "explicit_constraints": [],
            "wants_flight": False,
            "confidence": 0.75,
        },
        "dynamic_shop_pool": [],
        "retrieval_history": [],
    }


def _dummy_shop(name: str = "TestShop", tags: list[str] | None = None) -> ShopProfile:
    return ShopProfile(
        name=name,
        google_rating=4.2,
        trust_score=0.8,
        tags=tags or ["ramen", "main_meal", "lunch"],
        booking_type=BookingType.NONE,
        queue_strategy=QueueStrategy.PHYSICAL_LINE,
        close_time="21:00",
        last_call_offset=30,
        is_cash_only=False,
        sns_handle="",
        region="jp",
    )


# ---------------------------------------------------------------------------
# 1. LLM is called with RETRIEVAL_REASONING task type
# ---------------------------------------------------------------------------

class TestLLMCalled:
    def test_retriever_calls_llm(self) -> None:
        """RetrieverAgent.run() must invoke the LLM for candidate reasoning."""
        router = _mock_router()

        with patch("agents.retriever.NearbySearchTool") as mock_nearby:
            mock_nearby.return_value.search_places.return_value = []
            agent = RetrieverAgent(llm_router=router)
            report = agent.run(_minimal_state())

        router.complete.assert_called_once()
        call_args = router.complete.call_args
        task_arg = call_args[0][0] if call_args[0] else call_args[1].get("task")
        assert task_arg == TaskType.RETRIEVAL_REASONING

    def test_llm_notes_appear_in_report(self) -> None:
        """Notes returned by the LLM land in RetrievalReport.notes."""
        expected_note = "燃えよ麺助: top ramen pick for lunch."
        router = _mock_router(notes=[expected_note])

        with patch("agents.retriever.NearbySearchTool") as mock_nearby:
            mock_nearby.return_value.search_places.return_value = []
            agent = RetrieverAgent(llm_router=router)
            report = agent.run(_minimal_state())

        assert expected_note in report.notes

    def test_llm_receives_candidate_info(self) -> None:
        """The prompt sent to the LLM mentions the city from intent."""
        router = _mock_router()

        with patch("agents.retriever.NearbySearchTool") as mock_nearby:
            mock_nearby.return_value.search_places.return_value = []
            agent = RetrieverAgent(llm_router=router)
            agent.run(_minimal_state(city="台北", region="tw"))

        _, kwargs = router.complete.call_args if router.complete.call_args[0] else (None, router.complete.call_args[1])
        messages = router.complete.call_args[0][1] if router.complete.call_args[0] else router.complete.call_args[1].get("messages", [])
        full_text = " ".join(m.get("content", "") for m in messages)
        assert "台北" in full_text


# ---------------------------------------------------------------------------
# 2. Gaps report missing city catalog
# ---------------------------------------------------------------------------

class TestGapDetection:
    def test_unknown_city_triggers_catalog_gap(self) -> None:
        """When intent city has no dedicated catalog, gaps must mention it."""
        router = _mock_router(gaps=[])

        with patch("agents.retriever.NearbySearchTool") as mock_nearby:
            mock_nearby.return_value.search_places.return_value = []
            with patch("agents.retriever.load_shop_catalog", return_value=[]):
                agent = RetrieverAgent(llm_router=router)
                report = agent.run(_minimal_state(city="沖繩", region="jp", query="沖繩三天美食"))

        assert report.gaps, "Expected at least one gap for unknown city"
        all_gaps = " ".join(report.gaps)
        assert "沖繩" in all_gaps or "catalog" in all_gaps.lower() or "dedicated" in all_gaps.lower()

    def test_tokyo_catalog_empty_triggers_gap(self) -> None:
        """「東京三天美食」 with empty Tokyo catalog → gap about Tokyo."""
        router = _mock_router(gaps=["Tokyo seed catalog is sparse"])

        with patch("agents.retriever.NearbySearchTool") as mock_nearby:
            mock_nearby.return_value.search_places.return_value = []
            # Mock load_shop_catalog to return empty for tokyo
            def fake_catalog(name: str) -> list:
                return []  # all catalogs empty

            with patch("agents.retriever.load_shop_catalog", side_effect=fake_catalog):
                agent = RetrieverAgent(llm_router=router)
                report = agent.run(
                    _minimal_state(city="東京", region="jp", query="東京三天美食")
                )

        # Either structural gap (empty catalog) or LLM gap should mention Tokyo
        all_gap_text = " ".join(report.gaps)
        assert report.gaps, "Expected gaps for empty Tokyo catalog"
        # The structural detector should flag empty/no catalog
        has_tokyo_or_catalog_mention = (
            "東京" in all_gap_text
            or "tokyo" in all_gap_text.lower()
            or "catalog" in all_gap_text.lower()
            or "seed" in all_gap_text.lower()
        )
        assert has_tokyo_or_catalog_mention

    def test_uncovered_meal_slot_triggers_gap(self) -> None:
        """No shops covering 'late_night' → structural gap."""
        gaps = _detect_structural_gaps(
            city="京都",
            region="jp",
            meal_slots=["late_night"],
            category_tags=[],
            seed_shops=[_dummy_shop("DayShop", tags=["lunch", "main_meal"])],
            dynamic_shops=[],
        )
        assert any("late_night" in g or "late night" in g.lower() for g in gaps), \
            f"Expected late_night gap, got: {gaps}"

    def test_uncovered_category_tag_triggers_gap(self) -> None:
        """Requested tag 'sushi' not in any candidate → gap."""
        gaps = _detect_structural_gaps(
            city="台北",
            region="tw",
            meal_slots=[],
            category_tags=["sushi"],
            seed_shops=[_dummy_shop("RamenShop", tags=["ramen", "main_meal"])],
            dynamic_shops=[],
        )
        assert any("sushi" in g for g in gaps), f"Expected sushi gap, got: {gaps}"

    def test_has_dedicated_catalog(self) -> None:
        assert _has_dedicated_catalog("台北") is True
        assert _has_dedicated_catalog("京都") is True
        assert _has_dedicated_catalog("東京") is True
        assert _has_dedicated_catalog("沖繩") is False
        assert _has_dedicated_catalog("札幌") is False
        assert _has_dedicated_catalog("") is False

    def test_no_gap_when_catalog_exists(self) -> None:
        """Known cities (台北, 京都, 東京) don't trigger a 'no catalog' gap."""
        for city in ("台北", "京都", "東京"):
            gaps = _detect_structural_gaps(
                city=city,
                region="jp" if city != "台北" else "tw",
                meal_slots=[],
                category_tags=[],
                seed_shops=[_dummy_shop()],  # non-empty
                dynamic_shops=[],
            )
            catalog_gaps = [g for g in gaps if "catalog" in g.lower() and "dedicated" in g.lower()]
            assert not catalog_gaps, f"Unexpected catalog gap for {city}: {gaps}"


# ---------------------------------------------------------------------------
# 3. Retriever does NOT return a final itinerary
# ---------------------------------------------------------------------------

class TestRetrieverOutputContract:
    def test_report_has_no_final_itinerary(self) -> None:
        """RetrievalReport must not contain a 'final_itinerary' field."""
        router = _mock_router()

        with patch("agents.retriever.NearbySearchTool") as mock_nearby:
            mock_nearby.return_value.search_places.return_value = []
            agent = RetrieverAgent(llm_router=router)
            report = agent.run(_minimal_state())

        assert not hasattr(report, "final_itinerary")
        d = report.as_dict()
        assert "final_itinerary" not in d

    def test_report_has_required_fields(self) -> None:
        router = _mock_router()

        with patch("agents.retriever.NearbySearchTool") as mock_nearby:
            mock_nearby.return_value.search_places.return_value = []
            agent = RetrieverAgent(llm_router=router)
            report = agent.run(_minimal_state())

        assert isinstance(report.candidates, list)
        assert isinstance(report.notes, list)
        assert isinstance(report.gaps, list)
        assert isinstance(report.city, str)
        assert isinstance(report.region, str)

    def test_notes_are_strings_not_itinerary(self) -> None:
        """Notes should be short candidate evaluations, not full itineraries."""
        long_fake_itinerary = "Day 1: 早餐... 午餐... 晚餐... Day 2: ..."
        router = MagicMock()
        router.complete.return_value = _llm_response(
            notes=["燃えよ麺助: great ramen for lunch."],
            gaps=[],
        )

        with patch("agents.retriever.NearbySearchTool") as mock_nearby:
            mock_nearby.return_value.search_places.return_value = []
            agent = RetrieverAgent(llm_router=router)
            report = agent.run(_minimal_state())

        # Notes should not contain multi-day itinerary patterns
        for note in report.notes:
            assert "Day 1" not in note and "Day 2" not in note


# ---------------------------------------------------------------------------
# 4. as_dict() is JSON-serialisable
# ---------------------------------------------------------------------------

class TestAsDictSerialisation:
    def test_as_dict_json_serialisable(self) -> None:
        router = _mock_router()

        with patch("agents.retriever.NearbySearchTool") as mock_nearby:
            mock_nearby.return_value.search_places.return_value = []
            agent = RetrieverAgent(llm_router=router)
            report = agent.run(_minimal_state())

        d = report.as_dict()
        serialised = json.dumps(d, ensure_ascii=False)  # must not raise
        recovered = json.loads(serialised)
        assert "city" in recovered
        assert "candidate_names" in recovered
        assert "notes" in recovered
        assert "gaps" in recovered

    def test_as_dict_contains_expected_keys(self) -> None:
        report = RetrievalReport(
            city="京都",
            region="jp",
            candidates=[_dummy_shop("TestShop")],
            notes=["TestShop: decent ramen."],
            gaps=["No late-night option."],
            seed_count=1,
            dynamic_count=0,
            query="京都ランチ",
        )
        d = report.as_dict()
        assert d["city"] == "京都"
        assert d["region"] == "jp"
        assert "TestShop" in d["candidate_names"]
        assert d["seed_count"] == 1
        assert d["dynamic_count"] == 0
        assert d["query"] == "京都ランチ"


# ---------------------------------------------------------------------------
# 5. Empty candidate pool
# ---------------------------------------------------------------------------

class TestEmptyPool:
    def test_empty_pool_produces_gap_not_crash(self) -> None:
        """When both seed and dynamic return nothing, gaps must describe the problem."""
        router = _mock_router(notes=[], gaps=["No candidates found at all."])

        with patch("agents.retriever.NearbySearchTool") as mock_nearby:
            mock_nearby.return_value.search_places.return_value = []
            with patch("agents.retriever.load_shop_catalog", return_value=[]):
                agent = RetrieverAgent(llm_router=router)
                report = agent.run(_minimal_state(city="虛構市", region="jp"))

        assert isinstance(report, RetrievalReport)
        assert report.gaps  # must have at least one gap
        assert report.candidates == []  # no candidates is fine


# ---------------------------------------------------------------------------
# 6. LLM failure graceful fallback
# ---------------------------------------------------------------------------

class TestLLMFailure:
    def test_llm_exception_does_not_crash_retriever(self) -> None:
        router = MagicMock()
        router.complete.side_effect = RuntimeError("network failure")

        with patch("agents.retriever.NearbySearchTool") as mock_nearby:
            mock_nearby.return_value.search_places.return_value = []
            agent = RetrieverAgent(llm_router=router)
            report = agent.run(_minimal_state())

        assert isinstance(report, RetrievalReport)
        assert report.notes == []
        assert any("failed" in g.lower() or "LLM" in g for g in report.gaps)

    def test_llm_bad_json_produces_gap(self) -> None:
        """Unparseable LLM response adds a gap instead of crashing."""
        router = MagicMock()
        router.complete.return_value = LLMResponse(
            content="this is not json",
            model_used="mock",
            tokens_in=1,
            tokens_out=1,
            latency_ms=10,
            cost_usd=0.0,
        )

        with patch("agents.retriever.NearbySearchTool") as mock_nearby:
            mock_nearby.return_value.search_places.return_value = []
            agent = RetrieverAgent(llm_router=router)
            report = agent.run(_minimal_state())

        assert report.notes == []
        assert any("parsed" in g.lower() or "LLM" in g for g in report.gaps)


# ---------------------------------------------------------------------------
# 7. NearbySearchTool failure absorbed
# ---------------------------------------------------------------------------

class TestDynamicSearchFailure:
    def test_nearby_tool_exception_absorbed(self) -> None:
        """If NearbySearchTool raises, retriever still returns seed-only results."""
        router = _mock_router()

        with patch("agents.retriever.NearbySearchTool") as mock_nearby:
            mock_nearby.return_value.search_places.side_effect = ConnectionError("no internet")
            agent = RetrieverAgent(llm_router=router)
            report = agent.run(_minimal_state())

        # Should not raise; seed catalog should still be present
        assert isinstance(report, RetrievalReport)
        assert report.dynamic_count == 0


# ---------------------------------------------------------------------------
# 8. node_retriever wires report into state
# ---------------------------------------------------------------------------

import importlib.util as _importlib_util

_LANGGRAPH_AVAILABLE = _importlib_util.find_spec("langgraph") is not None


@pytest.mark.skipif(not _LANGGRAPH_AVAILABLE, reason="langgraph not installed in this environment")
class TestNodeRetriever:
    @pytest.mark.asyncio
    async def test_node_retriever_populates_retrieval_history(self) -> None:
        """node_retriever should append a RetrievalReport dict to state."""
        from agent import node_retriever

        state = {
            "query": "京都ランチ",
            "intent": {
                "city": "京都",
                "region": "jp",
                "meal_slots": ["lunch"],
                "category_tags": [],
                "dietary_hints": None,
                "mode": "balanced",
                "explicit_constraints": [],
                "wants_flight": False,
                "confidence": 0.8,
            },
            "dynamic_shop_pool": [],
            "retrieval_history": [],
            "research_log": [],
            "transit_audit": [],
        }

        mock_report_dict = {
            "city": "京都",
            "region": "jp",
            "query": "京都ランチ",
            "candidate_names": ["燃えよ麺助"],
            "notes": ["燃えよ麺助: good."],
            "gaps": [],
            "seed_count": 1,
            "dynamic_count": 0,
        }

        with patch("agent.RetrieverAgent") as MockAgent:
            mock_instance = MagicMock()
            mock_report = MagicMock()
            mock_report.as_dict.return_value = mock_report_dict
            mock_report.candidates = []
            mock_report.city = "京都"
            mock_report.seed_count = 1
            mock_report.dynamic_count = 0
            mock_report.notes = ["燃えよ麺助: good."]
            mock_report.gaps = []
            mock_instance.arun = AsyncMock(return_value=mock_report)
            MockAgent.return_value = mock_instance

            result = await node_retriever(state)

        assert len(result["retrieval_history"]) == 1
        assert result["retrieval_history"][0]["city"] == "京都"
        mock_instance.arun.assert_called_once()
