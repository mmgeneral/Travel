"""Tests for SynthesizerAgent and SynthesisReport.

Coverage
--------
* Output contract: three mandatory sections, freshness in [0,1].
* LLM is called with TaskType.SYNTHESIS.
* Heuristic fallback when LLM fails.
* Empty history short-circuit.
* discussion_freshness goes low when rounds repeat the same shops.
* node_synthesizer integration: writes to synthesis_history, logs transit_audit.
* JSON round-trip via SynthesisReport.as_dict().
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from agents.synthesizer import (
    SynthesizerAgent,
    SynthesisReport,
    _compute_freshness_heuristic,
    _parse_synthesis_response,
    _SynthesisSchema,
)
from llm_router import LLMResponse, TaskType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _mock_router(content: str = "", *, raise_exc: Exception | None = None) -> MagicMock:
    router = MagicMock()
    if raise_exc is not None:
        router.complete.side_effect = raise_exc
    else:
        router.complete.return_value = LLMResponse(
            content=content or _default_synthesis_json(),
            model_used="gemini-2.0-flash",
            tokens_in=120,
            tokens_out=80,
            latency_ms=400,
            cost_usd=0.0001,
        )
    return router


def _default_synthesis_json() -> str:
    return json.dumps({
        "consensus": ["三家店都有高 Google 評分（4.2+）", "Critic 同意挑選策略以在地店優先"],
        "unresolved": ["青島東路豆漿大王的早餐時段是否與行程衝突"],
        "frontier": ["缺少素食友善選項", "東京催眠拉麵尚未搜尋"],
        "discussion_freshness": 0.72,
    })


def _retrieval_round(candidates: list[str], gaps: list[str] | None = None) -> dict:
    return {
        "candidates": [{"name": c} for c in candidates],
        "notes": [f"Note about {c}" for c in candidates[:2]],
        "gaps": gaps or [],
    }


def _critique_round(
    verdict: str = "satisfied",
    accepted: list[str] | None = None,
    rejected: list[tuple[str, str]] | None = None,
    requests: list[str] | None = None,
) -> dict:
    return {
        "verdict": verdict,
        "accepted": [{"name": a} for a in (accepted or [])],
        "rejected_with_reason": list(rejected or []),
        "requests_for_retriever": list(requests or []),
        "llm_analysis": "Test critique analysis.",
    }


def _state_with_history(
    retrieval_rounds: list[list[str]],
    critique_rounds: list[dict] | None = None,
    gaps_per_round: list[list[str]] | None = None,
) -> dict:
    retrieval_history = [
        _retrieval_round(names, (gaps_per_round or [[]] * len(retrieval_rounds))[i])
        for i, names in enumerate(retrieval_rounds)
    ]
    critique_history = critique_rounds or [_critique_round()]
    return {
        "retrieval_history": retrieval_history,
        "critique_history": critique_history,
        "intent": {"city": "台北", "region": "tw", "category_tags": ["ramen"]},
        "synthesis_history": [],
        "transit_audit": [],
    }


# ---------------------------------------------------------------------------
# 1. Output contract
# ---------------------------------------------------------------------------

class TestOutputContract:
    def test_report_has_three_sections(self):
        router = _mock_router()
        agent = SynthesizerAgent(llm_router=router)
        state = _state_with_history([["店A", "店B", "店C"]])
        report = agent.run(state)

        assert isinstance(report, SynthesisReport)
        assert isinstance(report.consensus, list) and len(report.consensus) >= 1
        assert isinstance(report.unresolved, list)
        assert isinstance(report.frontier, list)

    def test_freshness_in_unit_interval(self):
        router = _mock_router()
        agent = SynthesizerAgent(llm_router=router)
        state = _state_with_history([["店A", "店B"]])
        report = agent.run(state)
        assert 0.0 <= report.discussion_freshness <= 1.0

    def test_round_counts_match_history(self):
        router = _mock_router()
        agent = SynthesizerAgent(llm_router=router)
        state = _state_with_history([["A", "B"], ["C", "D"]])
        state["critique_history"] = [_critique_round("request_more"), _critique_round("satisfied")]
        report = agent.run(state)
        assert report.retrieval_rounds == 2
        assert report.critique_rounds == 2

    def test_as_dict_is_json_serialisable(self):
        router = _mock_router()
        agent = SynthesizerAgent(llm_router=router)
        state = _state_with_history([["店A"]])
        report = agent.run(state)
        d = report.as_dict()
        dumped = json.dumps(d)
        loaded = json.loads(dumped)
        assert "consensus" in loaded
        assert "discussion_freshness" in loaded
        assert isinstance(loaded["discussion_freshness"], float)


# ---------------------------------------------------------------------------
# 2. LLM call routing
# ---------------------------------------------------------------------------

class TestLLMRouting:
    def test_calls_synthesis_task_type(self):
        router = _mock_router()
        agent = SynthesizerAgent(llm_router=router)
        state = _state_with_history([["A", "B"]])
        agent.run(state)
        router.complete.assert_called_once()
        call_args = router.complete.call_args
        assert call_args[0][0] == TaskType.SYNTHESIS

    def test_messages_contain_transcript_context(self):
        router = _mock_router()
        agent = SynthesizerAgent(llm_router=router)
        state = _state_with_history([["青島東路豆漿大王", "永康牛肉麵"]])
        agent.run(state)
        messages = router.complete.call_args[0][1]
        full_text = " ".join(m["content"] for m in messages)
        assert "青島東路豆漿大王" in full_text or "Retrieval round" in full_text

    def test_does_not_call_retriever_or_critic(self):
        router = _mock_router()
        agent = SynthesizerAgent(llm_router=router)
        state = _state_with_history([["A"]])
        with patch("agents.synthesizer.SynthesizerAgent._extract_frontier_heuristic") as mock_f:
            mock_f.return_value = []
            agent.run(state)
        # Only one LLM call (synthesis); no retriever/critic agents involved
        assert router.complete.call_count == 1


# ---------------------------------------------------------------------------
# 3. Heuristic fallback on LLM failure
# ---------------------------------------------------------------------------

class TestLLMFallback:
    def test_returns_report_even_when_llm_fails(self):
        router = _mock_router(raise_exc=RuntimeError("LLM down"))
        agent = SynthesizerAgent(llm_router=router)
        state = _state_with_history([["A", "B"]])
        report = agent.run(state)
        assert isinstance(report, SynthesisReport)
        assert 0.0 <= report.discussion_freshness <= 1.0

    def test_fallback_surfaces_retriever_gaps(self):
        router = _mock_router(raise_exc=RuntimeError("timeout"))
        agent = SynthesizerAgent(llm_router=router)
        state = _state_with_history(
            [["A"]],
            gaps_per_round=[["Missing vegan options", "No Tokyo catalog"]],
        )
        report = agent.run(state)
        all_text = " ".join(report.frontier)
        assert "Missing vegan options" in all_text or "No Tokyo catalog" in all_text

    def test_fallback_surfaces_latest_requests(self):
        router = _mock_router(raise_exc=ConnectionError("network"))
        agent = SynthesizerAgent(llm_router=router)
        state = _state_with_history([["A"]])
        state["critique_history"] = [
            _critique_round("request_more", requests=["需要更多素食選項", "要找地頭蛇店"]),
        ]
        report = agent.run(state)
        # unresolved should surface the last request_more requests
        all_text = " ".join(report.unresolved)
        assert "素食" in all_text or "地頭蛇" in all_text


# ---------------------------------------------------------------------------
# 4. Empty history short-circuit
# ---------------------------------------------------------------------------

class TestEmptyHistory:
    def test_empty_history_does_not_call_llm(self):
        router = _mock_router()
        agent = SynthesizerAgent(llm_router=router)
        state = {"retrieval_history": [], "critique_history": [], "intent": {}, "synthesis_history": [], "transit_audit": []}
        report = agent.run(state)
        router.complete.assert_not_called()
        assert report.discussion_freshness == 1.0

    def test_empty_history_returns_valid_report(self):
        router = _mock_router()
        agent = SynthesizerAgent(llm_router=router)
        state = {"retrieval_history": [], "critique_history": [], "intent": {}, "synthesis_history": [], "transit_audit": []}
        report = agent.run(state)
        assert isinstance(report.consensus, list)
        assert isinstance(report.frontier, list)


# ---------------------------------------------------------------------------
# 5. discussion_freshness heuristic
# ---------------------------------------------------------------------------

class TestFreshnessHeuristic:
    def test_first_round_is_fresh(self):
        rh = [_retrieval_round(["A", "B", "C"])]
        f = _compute_freshness_heuristic(rh, [])
        assert f > 0.5, f"Expected freshness > 0.5 for first round, got {f}"

    def test_identical_rounds_produce_low_freshness(self):
        same_candidates = ["永康牛肉麵", "青島東路豆漿大王", "欣葉"]
        rh = [_retrieval_round(same_candidates) for _ in range(3)]
        ch = [_critique_round("request_more") for _ in range(3)]
        f = _compute_freshness_heuristic(rh, ch)
        assert f < 0.4, f"Expected freshness < 0.4 for identical rounds, got {f}"

    def test_new_candidates_produce_high_freshness(self):
        rh = [
            _retrieval_round(["A", "B", "C"]),
            _retrieval_round(["D", "E", "F"]),
            _retrieval_round(["G", "H", "I"]),
        ]
        f = _compute_freshness_heuristic(rh, [])
        assert f > 0.6, f"Expected freshness > 0.6 for fully-new rounds, got {f}"

    def test_llm_freshness_low_when_repeating(self):
        """When LLM returns high freshness but heuristic is low, blend keeps it moderate."""
        llm_json = json.dumps({
            "consensus": ["Good shops"],
            "unresolved": [],
            "frontier": [],
            "discussion_freshness": 0.9,  # LLM overestimates
        })
        router = _mock_router(content=llm_json)
        agent = SynthesizerAgent(llm_router=router)
        same = ["A", "B", "C"]
        rh = [_retrieval_round(same) for _ in range(3)]
        state = {
            "retrieval_history": rh,
            "critique_history": [_critique_round("request_more")] * 3,
            "intent": {},
            "synthesis_history": [],
            "transit_audit": [],
        }
        report = agent.run(state)
        # Blend: 0.9 * 0.7 + heuristic(<0.4) * 0.3 → roughly 0.63 + 0.12 = 0.75 max
        # But heuristic for identical rounds is ~0 overlap → freshness ~0
        # Final should be below 0.8
        assert report.discussion_freshness < 0.85


# ---------------------------------------------------------------------------
# 6. node_synthesizer integration
# ---------------------------------------------------------------------------

class TestNodeSynthesizer:
    def test_node_writes_to_synthesis_history(self):
        from agent import node_synthesizer

        router = _mock_router()
        state: dict = _state_with_history([["A", "B"]])

        with patch("agent.SynthesizerAgent") as MockSA:
            mock_instance = MagicMock()
            MockSA.return_value = mock_instance
            mock_instance.run.return_value = SynthesisReport(
                consensus=["Consensus point"],
                unresolved=[],
                frontier=["Gap 1"],
                discussion_freshness=0.6,
            )
            updated = node_synthesizer(state)

        assert len(updated["synthesis_history"]) == 1
        entry = updated["synthesis_history"][0]
        assert entry["consensus"] == ["Consensus point"]
        assert entry["discussion_freshness"] == 0.6

    def test_node_logs_to_transit_audit(self):
        from agent import node_synthesizer

        state: dict = _state_with_history([["A"]])

        with patch("agent.SynthesizerAgent") as MockSA:
            mock_instance = MagicMock()
            MockSA.return_value = mock_instance
            mock_instance.run.return_value = SynthesisReport(
                consensus=["X"],
                unresolved=[],
                frontier=[],
                discussion_freshness=0.5,
            )
            updated = node_synthesizer(state)

        audit_text = " ".join(updated.get("transit_audit", []))
        assert "synthesis_complete" in audit_text

    def test_node_appends_not_overwrites(self):
        from agent import node_synthesizer

        state: dict = _state_with_history([["A"]])
        state["synthesis_history"] = [{"consensus": ["pre-existing"]}]

        with patch("agent.SynthesizerAgent") as MockSA:
            mock_instance = MagicMock()
            MockSA.return_value = mock_instance
            mock_instance.run.return_value = SynthesisReport(
                consensus=["New consensus"],
                unresolved=[],
                frontier=[],
                discussion_freshness=0.5,
            )
            updated = node_synthesizer(state)

        assert len(updated["synthesis_history"]) == 2
        assert updated["synthesis_history"][0]["consensus"] == ["pre-existing"]
        assert updated["synthesis_history"][1]["consensus"] == ["New consensus"]

    def test_synthesizer_does_not_write_retrieval_history(self):
        from agent import node_synthesizer

        state: dict = _state_with_history([["A"]])
        original_rh = list(state["retrieval_history"])

        with patch("agent.SynthesizerAgent") as MockSA:
            mock_instance = MagicMock()
            MockSA.return_value = mock_instance
            mock_instance.run.return_value = SynthesisReport(
                consensus=["X"], unresolved=[], frontier=[], discussion_freshness=0.5
            )
            updated = node_synthesizer(state)

        assert updated["retrieval_history"] == original_rh

    def test_synthesizer_does_not_write_critique_history(self):
        from agent import node_synthesizer

        state: dict = _state_with_history([["A"]])
        original_ch = list(state["critique_history"])

        with patch("agent.SynthesizerAgent") as MockSA:
            mock_instance = MagicMock()
            MockSA.return_value = mock_instance
            mock_instance.run.return_value = SynthesisReport(
                consensus=["X"], unresolved=[], frontier=[], discussion_freshness=0.5
            )
            updated = node_synthesizer(state)

        assert updated["critique_history"] == original_ch


# ---------------------------------------------------------------------------
# 7. JSON parsing robustness
# ---------------------------------------------------------------------------

class TestParseSynthesisResponse:
    def test_parses_clean_json(self):
        raw = json.dumps({
            "consensus": ["點 A"],
            "unresolved": ["點 B"],
            "frontier": ["缺口 C"],
            "discussion_freshness": 0.65,
        })
        result = _parse_synthesis_response(raw)
        assert result.consensus == ["點 A"]
        assert result.discussion_freshness == pytest.approx(0.65)

    def test_strips_markdown_fences(self):
        raw = "```json\n{\"consensus\":[\"X\"],\"unresolved\":[],\"frontier\":[],\"discussion_freshness\":0.5}\n```"
        result = _parse_synthesis_response(raw)
        assert result.consensus == ["X"]

    def test_invalid_json_returns_defaults(self):
        result = _parse_synthesis_response("not valid JSON at all")
        assert isinstance(result, _SynthesisSchema)
        assert result.consensus == []

    def test_freshness_clamped_to_unit_interval(self):
        raw = json.dumps({
            "consensus": [],
            "unresolved": [],
            "frontier": [],
            "discussion_freshness": 5.0,  # out of range
        })
        result = _parse_synthesis_response(raw)
        assert result.discussion_freshness == 1.0

    def test_string_sections_coerced_to_list(self):
        raw = json.dumps({
            "consensus": "single string consensus",
            "unresolved": [],
            "frontier": [],
            "discussion_freshness": 0.5,
        })
        result = _parse_synthesis_response(raw)
        assert isinstance(result.consensus, list)
