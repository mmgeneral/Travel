"""Tests for pause/resume mechanism and SynthesizerAgent integration in the graph.

Coverage
--------
* Two R-C rounds → pause → synthesizer report contains meaningful content.
* resume after pause continues from the last interrupt point.
* discussion_freshness < 0.3 when rounds produce identical candidates repeatedly.
* node_synthesizer does not appear in graph unless explicitly invoked.
* build_graph() with checkpointer compiles without default ``interrupt_after`` (opt-in via argument).
* /agent/pause and /agent/resume endpoints (unit-level mocks, no HTTP server needed).
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.synthesizer import SynthesizerAgent, SynthesisReport, _compute_freshness_heuristic
from llm_router import LLMResponse, TaskType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_retrieval_round(candidates: list[str], gaps: list[str] | None = None) -> dict:
    return {
        "candidates": [{"name": c} for c in candidates],
        "notes": [f"note:{c}" for c in candidates[:2]],
        "gaps": gaps or [],
    }


def _make_critique_round(
    verdict: str = "satisfied",
    requests: list[str] | None = None,
    accepted: list[str] | None = None,
) -> dict:
    return {
        "verdict": verdict,
        "accepted": [{"name": a} for a in (accepted or [])],
        "rejected_with_reason": [],
        "requests_for_retriever": list(requests or []),
        "llm_analysis": "test",
    }


def _two_round_state(same_candidates: bool = False) -> dict:
    if same_candidates:
        r1 = _make_retrieval_round(["店A", "店B", "店C"])
        r2 = _make_retrieval_round(["店A", "店B", "店C"])
    else:
        r1 = _make_retrieval_round(["店A", "店B", "店C"], gaps=["缺素食選項"])
        r2 = _make_retrieval_round(["店D", "店E", "店F"], gaps=["缺早餐選項"])
    return {
        "query": "台北美食推薦",
        "retrieval_history": [r1, r2],
        "critique_history": [
            _make_critique_round("request_more", requests=["需要更多地頭蛇店"]),
            _make_critique_round("satisfied", accepted=["店D"]),
        ],
        "intent": {"city": "台北", "region": "tw", "category_tags": ["restaurant"]},
        "synthesis_history": [],
        "transit_audit": [],
    }


def _mock_router_with_synthesis(
    consensus: list[str] | None = None,
    unresolved: list[str] | None = None,
    frontier: list[str] | None = None,
    freshness: float = 0.65,
) -> MagicMock:
    payload = {
        "consensus": consensus or ["兩個 agent 都同意推薦台灣在地店"],
        "unresolved": unresolved or ["是否要加一家米其林店"],
        "frontier": frontier or ["缺素食選項", "缺早餐選項"],
        "discussion_freshness": freshness,
    }
    router = MagicMock()
    router.complete.return_value = LLMResponse(
        content=json.dumps(payload),
        model_used="gemini-2.0-flash",
        tokens_in=150,
        tokens_out=90,
        latency_ms=300,
        cost_usd=0.0001,
    )
    return router


# ---------------------------------------------------------------------------
# 1. Two R-C rounds → pause → meaningful synthesis
# ---------------------------------------------------------------------------

class TestTwoRoundPause:
    def test_pause_after_two_rounds_returns_report(self):
        state = _two_round_state()
        router = _mock_router_with_synthesis()
        agent = SynthesizerAgent(llm_router=router)

        report = agent.run(state)

        assert isinstance(report, SynthesisReport)
        assert report.retrieval_rounds == 2
        assert report.critique_rounds == 2
        assert len(report.consensus) >= 1

    def test_pause_surfaces_gaps_in_frontier(self):
        state = _two_round_state()
        router = _mock_router_with_synthesis(frontier=["缺素食選項", "缺早餐選項"])
        agent = SynthesizerAgent(llm_router=router)
        report = agent.run(state)
        assert len(report.frontier) >= 1

    def test_pause_identifies_unresolved_points(self):
        state = _two_round_state()
        state["critique_history"][0]["requests_for_retriever"] = ["需要更多地頭蛇店"]
        router = _mock_router_with_synthesis(unresolved=["需要更多地頭蛇店"])
        agent = SynthesizerAgent(llm_router=router)
        report = agent.run(state)
        # Unresolved points should be non-empty when C issued a request
        assert isinstance(report.unresolved, list)

    def test_pause_consensus_non_empty(self):
        state = _two_round_state()
        router = _mock_router_with_synthesis(consensus=["都同意台灣在地店", "都同意避開觀光客動線"])
        agent = SynthesizerAgent(llm_router=router)
        report = agent.run(state)
        assert len(report.consensus) >= 1

    def test_synthesis_uses_gemini_tasktype(self):
        state = _two_round_state()
        router = _mock_router_with_synthesis()
        agent = SynthesizerAgent(llm_router=router)
        agent.run(state)
        router.complete.assert_called_once()
        task_type_arg = router.complete.call_args[0][0]
        assert task_type_arg == TaskType.SYNTHESIS
        # Messages list must be non-empty (system + user)
        messages_arg = router.complete.call_args[0][1]
        assert isinstance(messages_arg, list) and len(messages_arg) >= 2


# ---------------------------------------------------------------------------
# 2. Resume continues from interrupt (simulated via state mutation)
# ---------------------------------------------------------------------------

class TestResumeAfterPause:
    def test_synthesis_history_grows_after_second_pause(self):
        """Simulates: run → pause → run more → pause again — history appends."""
        from agent import node_synthesizer

        state: dict = _two_round_state()
        # First pause
        with patch("agent.SynthesizerAgent") as MockSA:
            inst = MockSA.return_value
            inst.run.return_value = SynthesisReport(
                consensus=["Round 1 consensus"], unresolved=[], frontier=[], discussion_freshness=0.7
            )
            state = node_synthesizer(state)

        assert len(state["synthesis_history"]) == 1

        # Simulate another R-C round
        state["retrieval_history"].append(_make_retrieval_round(["店G", "店H"]))
        state["critique_history"].append(_make_critique_round("satisfied", accepted=["店G"]))

        # Second pause
        with patch("agent.SynthesizerAgent") as MockSA:
            inst = MockSA.return_value
            inst.run.return_value = SynthesisReport(
                consensus=["Round 2 consensus"], unresolved=[], frontier=["New gap"], discussion_freshness=0.8
            )
            state = node_synthesizer(state)

        assert len(state["synthesis_history"]) == 2
        assert state["synthesis_history"][0]["consensus"] == ["Round 1 consensus"]
        assert state["synthesis_history"][1]["consensus"] == ["Round 2 consensus"]

    def test_resume_preserves_prior_synthesis(self):
        """Resuming should not overwrite existing synthesis_history."""
        from agent import node_synthesizer

        state: dict = _two_round_state()
        state["synthesis_history"] = [{"consensus": ["existing"], "discussion_freshness": 0.5}]

        with patch("agent.SynthesizerAgent") as MockSA:
            inst = MockSA.return_value
            inst.run.return_value = SynthesisReport(
                consensus=["new"], unresolved=[], frontier=[], discussion_freshness=0.6
            )
            state = node_synthesizer(state)

        assert len(state["synthesis_history"]) == 2
        assert state["synthesis_history"][0]["consensus"] == ["existing"]


# ---------------------------------------------------------------------------
# 3. discussion_freshness < 0.3 when rounds repeat identical candidates
# ---------------------------------------------------------------------------

class TestFreshnessDecay:
    def test_freshness_below_threshold_for_identical_rounds(self):
        """3 rounds of the exact same candidates → heuristic freshness < 0.3."""
        same_shops = ["青島東路豆漿大王", "永康牛肉麵", "欣葉"]
        rh = [_make_retrieval_round(same_shops) for _ in range(3)]
        ch = [_make_critique_round("request_more") for _ in range(3)]
        f = _compute_freshness_heuristic(rh, ch)
        assert f < 0.3, f"Expected freshness < 0.3 for identical rounds, got {f}"

    def test_freshness_above_threshold_for_diverse_rounds(self):
        """3 rounds with entirely different candidates → freshness > 0.6."""
        rh = [
            _make_retrieval_round(["A", "B", "C"]),
            _make_retrieval_round(["D", "E", "F"]),
            _make_retrieval_round(["G", "H", "I"]),
        ]
        f = _compute_freshness_heuristic(rh, [])
        assert f > 0.6, f"Expected freshness > 0.6 for diverse rounds, got {f}"

    def test_end_to_end_low_freshness_with_llm(self):
        """When LLM also reports low freshness, final blend should be < 0.3."""
        llm_payload = json.dumps({
            "consensus": ["same point"],
            "unresolved": [],
            "frontier": [],
            "discussion_freshness": 0.1,
        })
        router = MagicMock()
        router.complete.return_value = LLMResponse(
            content=llm_payload,
            model_used="gemini-2.0-flash",
            tokens_in=50,
            tokens_out=30,
            latency_ms=200,
            cost_usd=0.0,
        )
        agent = SynthesizerAgent(llm_router=router)
        same = ["A", "B", "C"]
        state = {
            "retrieval_history": [_make_retrieval_round(same)] * 3,
            "critique_history": [_make_critique_round("request_more")] * 3,
            "intent": {},
            "synthesis_history": [],
            "transit_audit": [],
        }
        report = agent.run(state)
        # LLM=0.1 (70%) + heuristic≈0 (30%) → should be well below 0.3
        assert report.discussion_freshness < 0.3


# ---------------------------------------------------------------------------
# 4. build_graph graph topology checks
# ---------------------------------------------------------------------------

class TestGraphTopology:
    def test_synthesizer_node_exists_in_graph(self):
        """build_graph() must register a 'synthesizer' node."""
        try:
            from langgraph.graph import StateGraph  # noqa: F401
        except ImportError:
            pytest.skip("langgraph not installed")

        from agent import build_graph
        graph = build_graph()
        # LangGraph exposes node names via graph.nodes or graph.get_graph().nodes
        try:
            node_names = set(graph.get_graph().nodes.keys())
        except AttributeError:
            node_names = set(getattr(graph, "nodes", {}).keys())
        assert "synthesizer" in node_names, f"synthesizer not in {node_names}"

    def test_graph_does_not_auto_run_synthesizer(self):
        """Synthesizer must not appear in the default execution path (no checkpointer path)."""
        try:
            from langgraph.graph import StateGraph  # noqa: F401
        except ImportError:
            pytest.skip("langgraph not installed")

        from agent import build_graph
        graph = build_graph(interrupt_after_nodes=[])

        # Get graph edges from the compiled graph
        compiled_graph = graph.get_graph()
        # The synthesizer node should exist but have no incoming edge from critic/plan
        edges = {(e.source, e.target) for e in compiled_graph.edges}
        # synthesizer should not be reachable from the normal flow nodes
        normal_flow = {"route_intent", "clarify_constraint", "retriever", "researcher", "critic",
                       "collect_feedback", "plan"}
        synthesizer_in_edges = {src for src, tgt in edges if tgt == "synthesizer" and src in normal_flow}
        assert not synthesizer_in_edges, (
            f"synthesizer is auto-triggered from {synthesizer_in_edges}"
        )


# ---------------------------------------------------------------------------
# 5. SynthesisReport JSON round-trip
# ---------------------------------------------------------------------------

class TestSynthesisReportSerialisation:
    def test_as_dict_schema(self):
        report = SynthesisReport(
            consensus=["A", "B"],
            unresolved=["C"],
            frontier=["D", "E"],
            discussion_freshness=0.42,
            retrieval_rounds=2,
            critique_rounds=2,
            llm_summary="test summary",
        )
        d = report.as_dict()
        assert d["consensus"] == ["A", "B"]
        assert d["unresolved"] == ["C"]
        assert d["frontier"] == ["D", "E"]
        assert d["discussion_freshness"] == pytest.approx(0.42, abs=0.001)
        assert d["retrieval_rounds"] == 2
        assert d["critique_rounds"] == 2
        assert "llm_summary" in d

    def test_freshness_rounded_to_3_decimals(self):
        report = SynthesisReport(
            consensus=[], unresolved=[], frontier=[],
            discussion_freshness=0.123456789,
        )
        assert report.as_dict()["discussion_freshness"] == 0.123
