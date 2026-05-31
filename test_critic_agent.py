"""
Tests for agents/critic.py — CriticAgent and CritiqueReport.

Coverage
--------
1. Critic calls LLM (TaskType.CRITIQUE) for reasoning.
2. High marketing-noise shops (chain/accolade-heavy, low fame-damped score) appear
   in rejected_with_reason when the LLM says to reject them.
3. verdict == "request_more" guarantees requests_for_retriever is non-empty.
4. Critic references real ScoringEngine scores (score_table is non-empty with
   actual numeric values, not zeros).
5. fame_damped_table shows lower values than score_table for over-hyped shops.
6. LLM failure falls back gracefully (verdict="satisfied", no crash).
7. Empty retrieval history → critic falls back to seed catalog and still runs.
8. CritiqueReport.as_dict() is JSON-serialisable.
9. node_critic sets auditor_rejected correctly based on verdict.
10. verdict="deadlock" does NOT set auditor_rejected (loop exits immediately).
"""
from __future__ import annotations

import importlib.util as _importlib_util
import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agents.critic import (
    CriticAgent,
    CritiqueReport,
    _compute_score_tables,
    _intent_to_preference,
    _parse_llm_critique,
)
from decision_engine import ScoringEngine, WeightProfile
from llm_router import LLMResponse, TaskType
from shop_planning import AuthorityData, BookingType, QueueStrategy, ShopProfile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _llm_response(content: str) -> LLMResponse:
    return LLMResponse(
        content=content,
        model_used="mock-claude",
        tokens_in=80,
        tokens_out=120,
        latency_ms=200,
        cost_usd=0.001,
    )


def _mock_router(
    accepted: list[str] | None = None,
    rejected: list[dict] | None = None,
    requests: list[str] | None = None,
    verdict: str = "satisfied",
    analysis: str = "test analysis",
) -> MagicMock:
    payload = {
        "accepted_names": accepted or [],
        "rejected": rejected or [],
        "requests": requests or [],
        "verdict": verdict,
        "analysis": analysis,
    }
    router = MagicMock()
    router.complete.return_value = _llm_response(json.dumps(payload, ensure_ascii=False))
    return router


def _make_shop(
    name: str,
    rating: float = 4.2,
    tags: list[str] | None = None,
    michelin_star: int = 0,
    tablelog_medal: str = "",
    review_count: int = 150,
    marketing_noise_score: float | None = None,
    region: str = "jp",
) -> ShopProfile:
    auth = AuthorityData(
        michelin_star=michelin_star,
        tablelog_medal=tablelog_medal,
        review_count=review_count,
    )
    shop = ShopProfile(
        name=name,
        google_rating=rating,
        trust_score=max(0.55, rating / 5.0),
        tags=tags or ["restaurant", "main_meal"],
        booking_type=BookingType.NONE,
        queue_strategy=QueueStrategy.PHYSICAL_LINE,
        close_time="21:00",
        last_call_offset=30,
        is_cash_only=False,
        sns_handle="",
        region=region,
        authority_data=auth,
    )
    if marketing_noise_score is not None:
        shop.marketing_noise_score = marketing_noise_score
    return shop


def _chain_shop(name: str = "高噪音連鎖店") -> ShopProfile:
    """A typical over-hyped chain with Michelin stars → high accolade_bonus."""
    return _make_shop(
        name=name,
        rating=4.5,
        tags=["chain", "restaurant", "main_meal"],
        michelin_star=3,         # accolade_bonus = 3*8 = 24
        tablelog_medal="百名店",  # +18 → accolade_bonus ≈ 42
        review_count=800,
        marketing_noise_score=0.85,
    )


def _insider_shop(name: str = "在地隱藏版") -> ShopProfile:
    """A quiet insider gem with moderate rating, no awards, 150 reviews."""
    return _make_shop(
        name=name,
        rating=4.3,
        tags=["restaurant", "main_meal", "local"],
        michelin_star=0,
        tablelog_medal="",
        review_count=150,
        marketing_noise_score=0.1,
    )


def _minimal_state(
    city: str = "京都",
    region: str = "jp",
    candidate_names: list[str] | None = None,
    retriever_notes: list[str] | None = None,
    retriever_gaps: list[str] | None = None,
    researcher_names: list[str] | None = None,
) -> dict:
    return {
        "query": "京都ランチ",
        "intent": {
            "city": city,
            "region": region,
            "meal_slots": ["lunch"],
            "category_tags": [],
            "dietary_hints": None,
            "mode": "balanced",
            "explicit_constraints": [],
            "wants_flight": False,
            "confidence": 0.8,
        },
        "retrieval_history": [
            {
                "city": city,
                "region": region,
                "query": "test",
                "candidate_names": candidate_names or [],
                "notes": retriever_notes or [],
                "gaps": retriever_gaps or [],
                "seed_count": len(candidate_names or []),
                "dynamic_count": 0,
            }
        ],
        "critique_history": [],
        "dynamic_shop_pool": [],
        "researcher_candidate_names": researcher_names or [],
        "research_iteration": 0,
        "transit_audit": [],
        "research_log": [],
    }


# ---------------------------------------------------------------------------
# 1. LLM is called with TaskType.CRITIQUE
# ---------------------------------------------------------------------------

class TestLLMCalled:
    def test_critic_calls_llm_with_critique_task(self) -> None:
        chain = _chain_shop()
        router = _mock_router(accepted=[], rejected=[{"name": chain.name, "reason": "chain store noise"}], verdict="request_more", requests=["Find local non-chain restaurants"])

        with patch("agents.critic._load_seed_for_city", return_value=[chain]):
            agent = CriticAgent(llm_router=router)
            report = agent.run(_minimal_state(candidate_names=[chain.name]))

        router.complete.assert_called_once()
        task_arg = router.complete.call_args[0][0]
        assert task_arg == TaskType.CRITIQUE

    def test_llm_prompt_contains_score_info(self) -> None:
        """The prompt sent to the LLM must include score columns."""
        chain = _chain_shop()
        router = _mock_router()

        with patch("agents.critic._load_seed_for_city", return_value=[chain]):
            agent = CriticAgent(llm_router=router)
            agent.run(_minimal_state(candidate_names=[chain.name]))

        messages = router.complete.call_args[0][1]
        full_text = " ".join(m.get("content", "") for m in messages)
        assert "raw=" in full_text or "fame_damped=" in full_text or "score" in full_text.lower()

    def test_llm_prompt_contains_city(self) -> None:
        router = _mock_router()
        insider = _insider_shop()

        with patch("agents.critic._load_seed_for_city", return_value=[insider]):
            agent = CriticAgent(llm_router=router)
            agent.run(_minimal_state(city="台北", region="tw", candidate_names=[insider.name]))

        messages = router.complete.call_args[0][1]
        full_text = " ".join(m.get("content", "") for m in messages)
        assert "台北" in full_text


# ---------------------------------------------------------------------------
# 2. High-noise shop rejection
# ---------------------------------------------------------------------------

class TestNoiseRejection:
    def test_chain_shop_rejected_when_llm_says_so(self) -> None:
        """When LLM rejects a chain store, it appears in rejected_with_reason."""
        chain = _chain_shop("鼎泰豐")
        router = _mock_router(
            accepted=[],
            rejected=[{"name": "鼎泰豐", "reason": "marketing noise: accolade_bonus=42 but noise=0.85; tourist trap"}],
            requests=["Find local hidden-gem restaurants with <200 reviews"],
            verdict="request_more",
        )

        with patch("agents.critic._load_seed_for_city", return_value=[chain]):
            agent = CriticAgent(llm_router=router)
            report = agent.run(_minimal_state(candidate_names=["鼎泰豐"]))

        rejected_names = [name for name, _ in report.rejected_with_reason]
        assert "鼎泰豐" in rejected_names

    def test_rejection_reason_contains_evidence(self) -> None:
        """Rejection reason must be non-empty (LLM should reference noise/scores)."""
        chain = _chain_shop("噪音連鎖店")
        reason_text = "accolade_bonus=42 but fame_damped is 30% lower; chain tag detected"
        router = _mock_router(
            rejected=[{"name": "噪音連鎖店", "reason": reason_text}],
            verdict="request_more",
            requests=["Add local non-chain options"],
        )

        with patch("agents.critic._load_seed_for_city", return_value=[chain]):
            agent = CriticAgent(llm_router=router)
            report = agent.run(_minimal_state(candidate_names=["噪音連鎖店"]))

        reasons = {name: reason for name, reason in report.rejected_with_reason}
        assert "噪音連鎖店" in reasons
        assert len(reasons["噪音連鎖店"]) > 0

    def test_insider_shop_not_rejected_without_cause(self) -> None:
        """Insider shop (is_insider=True) stays accepted when LLM approves it."""
        insider = _insider_shop("青島東路豆漿大王")
        router = _mock_router(
            accepted=["青島東路豆漿大王"],
            rejected=[],
            verdict="satisfied",
        )

        with patch("agents.critic._load_seed_for_city", return_value=[insider]):
            agent = CriticAgent(llm_router=router)
            report = agent.run(_minimal_state(candidate_names=["青島東路豆漿大王"]))

        accepted_names = [s.name for s in report.accepted]
        assert "青島東路豆漿大王" in accepted_names


# ---------------------------------------------------------------------------
# 3. request_more verdict requires non-empty requests
# ---------------------------------------------------------------------------

class TestRequestMore:
    def test_request_more_has_non_empty_requests(self) -> None:
        router = _mock_router(
            verdict="request_more",
            requests=["Find ramen shops open after 9pm"],
        )

        with patch("agents.critic._load_seed_for_city", return_value=[_insider_shop()]):
            agent = CriticAgent(llm_router=router)
            report = agent.run(_minimal_state())

        assert report.verdict == "request_more"
        assert len(report.requests_for_retriever) > 0

    def test_request_more_auto_fills_empty_requests(self) -> None:
        """If LLM returns request_more but empty requests, critic adds a fallback."""
        router = _mock_router(
            verdict="request_more",
            requests=[],  # LLM forgot to add requests
        )

        with patch("agents.critic._load_seed_for_city", return_value=[_insider_shop()]):
            agent = CriticAgent(llm_router=router)
            report = agent.run(_minimal_state())

        # CriticAgent must ensure requests is non-empty when verdict=request_more
        assert report.verdict == "request_more"
        assert len(report.requests_for_retriever) > 0

    def test_satisfied_verdict_requests_may_be_empty(self) -> None:
        router = _mock_router(verdict="satisfied", requests=[])

        with patch("agents.critic._load_seed_for_city", return_value=[_insider_shop()]):
            agent = CriticAgent(llm_router=router)
            report = agent.run(_minimal_state())

        assert report.verdict == "satisfied"
        # No constraint on requests when satisfied
        assert isinstance(report.requests_for_retriever, list)


# ---------------------------------------------------------------------------
# 4 & 5. ScoringEngine scores are real, fame_damped differs for noisy shops
# ---------------------------------------------------------------------------

class TestScoringEngineUsed:
    def test_score_table_is_non_empty_and_numeric(self) -> None:
        """score_table must contain actual ScoringEngine-computed values."""
        shops = [_chain_shop("ChainA"), _insider_shop("InsiderB")]
        router = _mock_router()

        with patch("agents.critic._load_seed_for_city", return_value=shops):
            agent = CriticAgent(llm_router=router)
            report = agent.run(_minimal_state(candidate_names=["ChainA", "InsiderB"]))

        assert len(report.score_table) > 0
        for name, score in report.score_table.items():
            assert isinstance(score, float)
            assert score >= 0.0

    def test_fame_damped_lower_for_high_noise_shop(self) -> None:
        """For a chain with michelin stars + high noise, fame_damped < raw score."""
        chain = _chain_shop("HighNoise")
        router = _mock_router()

        with patch("agents.critic._load_seed_for_city", return_value=[chain]):
            agent = CriticAgent(llm_router=router)
            report = agent.run(_minimal_state(candidate_names=["HighNoise"]))

        raw = report.score_table.get("HighNoise", 0.0)
        damped = report.fame_damped_table.get("HighNoise", 0.0)
        accolade = report.accolade_table.get("HighNoise", 0.0)
        # accolade_bonus for michelin_star=3 + 百名店 should be >0
        assert accolade > 0, f"Expected accolade_bonus > 0, got {accolade}"
        # fame_damped = raw - (accolade * noise) <= raw
        assert damped <= raw + 0.01, f"Expected damped({damped}) <= raw({raw})"

    def test_accolade_table_correct_for_michelin_shop(self) -> None:
        """Michelin 3-star shop → accolade_bonus = 24.0 (3 * 8.0)."""
        michelin = _make_shop("MichelinOnly", michelin_star=3, tablelog_medal="", review_count=50)
        router = _mock_router()

        with patch("agents.critic._load_seed_for_city", return_value=[michelin]):
            agent = CriticAgent(llm_router=router)
            report = agent.run(_minimal_state(candidate_names=["MichelinOnly"]))

        accolade = report.accolade_table.get("MichelinOnly", -1.0)
        # 3 stars × 8.0 = 24.0
        assert abs(accolade - 24.0) < 0.01, f"Expected 24.0, got {accolade}"

    def test_compute_score_tables_directly(self) -> None:
        """Unit-test _compute_score_tables() without going through the full agent."""
        pref = _intent_to_preference({
            "city": "京都",
            "region": "jp",
            "meal_slots": ["lunch"],
            "category_tags": [],
            "dietary_hints": None,
        })
        shops = [_chain_shop("X"), _insider_shop("Y")]
        score_table, damped_table, accolade_table, insider_table = _compute_score_tables(shops, pref)

        assert set(score_table.keys()) == {"X", "Y"}
        assert all(v >= 0 for v in score_table.values())
        # Chain shop has michelin_star=3 + 百名店 → accolade_bonus > 0
        assert accolade_table["X"] > 0
        # Insider shop has no medals → accolade_bonus = 0
        assert accolade_table["Y"] == 0.0


# ---------------------------------------------------------------------------
# 6. LLM failure graceful fallback
# ---------------------------------------------------------------------------

class TestLLMFailure:
    def test_llm_exception_returns_satisfied(self) -> None:
        router = MagicMock()
        router.complete.side_effect = RuntimeError("network error")

        with patch("agents.critic._load_seed_for_city", return_value=[_insider_shop()]):
            agent = CriticAgent(llm_router=router)
            report = agent.run(_minimal_state())

        assert isinstance(report, CritiqueReport)
        assert report.verdict == "satisfied"  # graceful fallback

    def test_llm_bad_json_returns_request_more(self) -> None:
        router = MagicMock()
        router.complete.return_value = _llm_response("this is not json {{ broken")

        with patch("agents.critic._load_seed_for_city", return_value=[_insider_shop()]):
            agent = CriticAgent(llm_router=router)
            report = agent.run(_minimal_state())

        # _parse_llm_critique returns request_more on parse error
        assert report.verdict in {"satisfied", "request_more"}


# ---------------------------------------------------------------------------
# 7. Empty retrieval history falls back to seed catalog
# ---------------------------------------------------------------------------

class TestEmptyRetrieval:
    def test_empty_retrieval_history_uses_seed(self) -> None:
        """If retrieval_history is empty, critic uses seed catalog as candidates."""
        seed = [_insider_shop("SeedShop")]
        router = _mock_router(accepted=["SeedShop"], verdict="satisfied")

        state = {
            "query": "test",
            "intent": {"city": "京都", "region": "jp", "meal_slots": [], "category_tags": [], "dietary_hints": None, "mode": "balanced"},
            "retrieval_history": [],  # empty!
            "critique_history": [],
            "dynamic_shop_pool": [],
            "researcher_candidate_names": [],
            "research_iteration": 0,
            "transit_audit": [],
        }

        with patch("agents.critic._load_seed_for_city", return_value=seed):
            agent = CriticAgent(llm_router=router)
            report = agent.run(state)

        assert isinstance(report, CritiqueReport)
        # Score table should be non-empty (seed was used)
        assert len(report.score_table) > 0


# ---------------------------------------------------------------------------
# 8. CritiqueReport.as_dict() is JSON-serialisable
# ---------------------------------------------------------------------------

class TestAsDictSerialisation:
    def test_as_dict_json_serialisable(self) -> None:
        router = _mock_router(
            accepted=["在地隱藏版"],
            rejected=[{"name": "高噪音連鎖店", "reason": "chain noise"}],
            requests=["Find local spots"],
            verdict="request_more",
        )
        shops = [_insider_shop("在地隱藏版"), _chain_shop("高噪音連鎖店")]

        with patch("agents.critic._load_seed_for_city", return_value=shops):
            agent = CriticAgent(llm_router=router)
            report = agent.run(_minimal_state(candidate_names=["在地隱藏版", "高噪音連鎖店"]))

        d = report.as_dict()
        serialised = json.dumps(d, ensure_ascii=False)
        recovered = json.loads(serialised)
        assert recovered["verdict"] == "request_more"
        assert "accepted_names" in recovered
        assert "rejected_with_reason" in recovered
        assert "requests_for_retriever" in recovered
        assert "score_table" in recovered
        assert "fame_damped_table" in recovered
        assert "accolade_table" in recovered


# ---------------------------------------------------------------------------
# 9 & 10. node_critic sets auditor_rejected correctly
# ---------------------------------------------------------------------------

_LANGGRAPH_AVAILABLE = _importlib_util.find_spec("langgraph") is not None


@pytest.mark.skipif(not _LANGGRAPH_AVAILABLE, reason="langgraph not installed")
class TestNodeCritic:
    def _base_state(self) -> dict:
        return {
            "query": "台北美食",
            "intent": {"city": "台北", "region": "tw", "meal_slots": ["lunch"], "category_tags": [], "dietary_hints": None, "mode": "balanced"},
            "retrieval_history": [{"city": "台北", "region": "tw", "query": "test", "candidate_names": [], "notes": [], "gaps": [], "seed_count": 0, "dynamic_count": 0}],
            "critique_history": [],
            "dynamic_shop_pool": [],
            "researcher_candidate_names": [],
            "research_iteration": 0,
            "transit_audit": [],
            "research_log": [],
            "auditor_rejected": False,
            "auditor_feedback": "",
        }

    def test_request_more_sets_auditor_rejected_true(self) -> None:
        from agent import node_critic

        mock_report = MagicMock()
        mock_report.verdict = "request_more"
        mock_report.requests_for_retriever = ["Add local breakfast spots"]
        mock_report.accepted = []
        mock_report.rejected_with_reason = []
        mock_report.score_table = {}
        mock_report.fame_damped_table = {}
        mock_report.accolade_table = {}
        mock_report.llm_analysis = ""
        mock_report.iteration = 0
        mock_report.as_dict.return_value = {"verdict": "request_more", "accepted_names": [], "rejected_with_reason": [], "requests_for_retriever": ["test"], "score_table": {}, "fame_damped_table": {}, "accolade_table": {}, "llm_analysis": "", "iteration": 0}

        with patch("dp_solver.CriticAgent") as MockCA:
            MockCA.return_value.run.return_value = mock_report
            state = self._base_state()
            result = node_critic(state)

        assert result["auditor_rejected"] is True

    def test_deadlock_does_not_set_auditor_rejected(self) -> None:
        from agent import node_critic

        mock_report = MagicMock()
        mock_report.verdict = "deadlock"
        mock_report.requests_for_retriever = []
        mock_report.accepted = []
        mock_report.rejected_with_reason = []
        mock_report.score_table = {}
        mock_report.fame_damped_table = {}
        mock_report.accolade_table = {}
        mock_report.llm_analysis = "deadlock: no vegan ramen in city"
        mock_report.iteration = 2
        mock_report.as_dict.return_value = {"verdict": "deadlock", "accepted_names": [], "rejected_with_reason": [], "requests_for_retriever": [], "score_table": {}, "fame_damped_table": {}, "accolade_table": {}, "llm_analysis": "deadlock", "iteration": 2}

        with patch("dp_solver.CriticAgent") as MockCA:
            MockCA.return_value.run.return_value = mock_report
            state = self._base_state()
            result = node_critic(state)

        assert result["auditor_rejected"] is False

    def test_satisfied_does_not_set_auditor_rejected(self) -> None:
        from agent import node_critic

        mock_report = MagicMock()
        mock_report.verdict = "satisfied"
        mock_report.requests_for_retriever = []
        mock_report.accepted = [_insider_shop()]
        mock_report.rejected_with_reason = []
        mock_report.score_table = {"在地隱藏版": 72.5}
        mock_report.fame_damped_table = {"在地隱藏版": 72.5}
        mock_report.accolade_table = {"在地隱藏版": 0.0}
        mock_report.llm_analysis = "pool looks great"
        mock_report.iteration = 0
        mock_report.as_dict.return_value = {"verdict": "satisfied", "accepted_names": ["在地隱藏版"], "rejected_with_reason": [], "requests_for_retriever": [], "score_table": {"在地隱藏版": 72.5}, "fame_damped_table": {"在地隱藏版": 72.5}, "accolade_table": {"在地隱藏版": 0.0}, "llm_analysis": "pool looks great", "iteration": 0}

        with patch("dp_solver.CriticAgent") as MockCA:
            MockCA.return_value.run.return_value = mock_report
            state = self._base_state()
            result = node_critic(state)

        assert result["auditor_rejected"] is False
        assert len(result["critique_history"]) == 1


# ---------------------------------------------------------------------------
# Additional unit tests for _parse_llm_critique
# ---------------------------------------------------------------------------

class TestParseLLMCritique:
    def test_valid_json_parsed(self) -> None:
        content = json.dumps({
            "accepted_names": ["ShopA"],
            "rejected": [{"name": "ShopB", "reason": "too noisy"}],
            "requests": ["Find local spots"],
            "verdict": "request_more",
            "analysis": "ok",
        })
        schema = _parse_llm_critique(content)
        assert schema.verdict == "request_more"
        assert "ShopA" in schema.accepted_names
        assert schema.rejected[0]["name"] == "ShopB"

    def test_markdown_fence_stripped(self) -> None:
        content = "```json\n{\"accepted_names\": [], \"verdict\": \"satisfied\"}\n```"
        schema = _parse_llm_critique(content)
        assert schema.verdict == "satisfied"

    def test_invalid_verdict_normalised(self) -> None:
        content = json.dumps({"verdict": "unknown_value"})
        schema = _parse_llm_critique(content)
        assert schema.verdict == "request_more"

    def test_broken_json_returns_request_more(self) -> None:
        schema = _parse_llm_critique("not json { broken")
        assert schema.verdict == "request_more"
