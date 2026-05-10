from __future__ import annotations
from catalog import (
    _build_shop_catalog,
    _build_shop_catalog_taipei,
    _build_shop_catalog_tokyo,
)
import copy
import json
import asyncio
import logging
import traceback
from contextlib import contextmanager
from langgraph.graph import END, StateGraph
from langchain_core.runnables.config import RunnableConfig
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import os
import re
from pathlib import Path
import sqlite3
import uuid

from debug_json import audit_json_line_as_text, debug_json as _dj
from zoneinfo import ZoneInfo
from typing import Any, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from duffel import DuffelService
from openai_completion_client import OpenAIChatCompletionClient
from saga import SagaEngine, SagaStep
from shop_catalog_io import load_shop_catalog
from shop_planning import (
    AuthorityData,
    BookingType,
    DietaryAxis,
    FlavorCategory,
    MockSnsProvider,
    MockTrafficProvider,
    NearbySearchTool,
    QueueStrategy,
    ReviewAnalyzer,
    ShopProfile,
    plan_shop_visit,
)
from graph_checkpoint_utils import extend_turn_checkpoint_in_state
from retrieval_service import (
    DietaryConstraints,
    catalog_stem_for_city,
    filter_shop_profiles_by_dietary_exclusions,
    filter_shop_profiles_by_excluded_shop_names,
    normalized_excluded_shop_names_from_intent,
    plan_excluded_frozenset,
    retrieve_seed_candidates,
)
from decision_engine import (
    DensityScanner,
    GraphBuilder,
    GraphEdge,
    GraphNode,
    HealthBudgetTracker,
    ScoringEngine,
    ItinerarySynthesizer,
    OptimizationMode,
    RankingEngine,
    RankedShop,
    SelectionStrategy,
    SpatioTemporalGraph,
    UserPreferenceLearner,
    UserMinefield,
    UserPreference,
    WeightProfile,
    choose_health_backup,
)

from intent_parser import intent_from_snapshot_dict as _intent_from_snapshot_dict
from intent_parser import parse_intent as _parse_intent
from tracing import trace_agent_stage
from llm_router import LLMRouter as _LLMRouter
from agents.retriever import RetrieverAgent
from agents.critic import CriticAgent
from agents.synthesizer import SynthesizerAgent, SynthesisReport
from observability import traced

_RAW_GRAPHBUILDER_BUILD = GraphBuilder.build_graph

def _user_negates_food_category_in_query(q_raw: str, category: str) -> bool:
    q = (q_raw or "").lower()
    if category == "ramen":
        return any(
            k in (q_raw or "") for k in ("不吃拉麵", "不吃拉面", "不要拉麵", "不要拉面", "忌拉麵", "忌拉面")
        ) or any(x in q for x in ("no ramen", "avoid ramen", "without ramen"))
    if category == "izakaya":
        return any(k in (q_raw or "") for k in ("不吃居酒屋", "不要居酒屋")) or "no izakaya" in q or "avoid izakaya" in q
    if category == "dessert":
        return ("不吃甜點" in (q_raw or "") or "不吃甜点" in (q_raw or "") or "不要甜點" in (q_raw or "")) or (
            "no dessert" in q or "avoid dessert" in q
        )
    return False


def _constraint_string_to_dietary_keys(constraint_raw: str) -> list[str]:
    s = (constraint_raw or "").strip()
    if not s:
        return []
    sl = s.lower().replace("-", "_")
    direct = ItinerarySynthesizer._canonical_dietary_constraint_key(sl)
    if direct:
        return [direct]
    keys: list[str] = []
    if "不吃拉麵" in s or "不吃拉面" in s or ("拉麵" in s and "不吃" in s) or ("拉面" in s and "不吃" in s):
        keys.append("no_ramen")
    if "不吃牛" in s or ("牛肉" in s and "不吃" in s):
        keys.append("no_beef")
    if "不吃豬" in s or "不吃猪" in s or ("豬" in s and "不吃" in s) or ("猪" in s and "不吃" in s) or "不吃豚" in s:
        keys.append("no_pork")
    return list(dict.fromkeys(keys))


def _dietary_keys_from_query_and_intent_signals(query_text: str) -> list[str]:
    q_raw = query_text or ""
    q = q_raw.lower()
    keys: list[str] = []
    if "vegan" in q or "純素" in q_raw:
        keys.append("vegan")
    elif "vegetarian" in q or "素食" in q_raw:
        keys.append("vegetarian")
    elif "pescatarian" in q:
        keys.append("pescatarian")
    if any(k in q_raw for k in ("不吃牛", "不吃牛肉")) or "no beef" in q or "avoid beef" in q:
        keys.append("no_beef")
    if any(k in q_raw for k in ("不吃拉麵", "不吃拉面")) or "no ramen" in q or "avoid ramen" in q:
        keys.append("no_ramen")
    if any(k in q_raw for k in ("不吃豬", "不吃猪", "不吃豚")) or "no pork" in q or "avoid pork" in q:
        keys.append("no_pork")
    return list(dict.fromkeys(keys))


def _build_plan_excluded_shop_tags(
    query_text: str,
    intent: dict | None,
    dietary_profile: dict | None,
) -> frozenset[str]:
    """Union of excluded shop tags from dietary_hints, profile ethics, explicit_constraints, and query."""
    _intent = intent or {}
    prof = dietary_profile or {}
    key_list: list[str] = []

    dh = _intent.get("dietary_hints")
    if dh is not None and str(dh).strip():
        for part in str(dh).split(","):
            k = ItinerarySynthesizer._canonical_dietary_constraint_key(part.strip())
            if k:
                key_list.append(k)

    pe = prof.get("ethics")
    if pe is not None and str(pe).strip():
        ek = ItinerarySynthesizer._canonical_dietary_constraint_key(str(pe).strip())
        if ek:
            key_list.append(ek)

    for ec in _intent.get("explicit_constraints") or []:
        key_list.extend(_constraint_string_to_dietary_keys(str(ec)))

    key_list.extend(_dietary_keys_from_query_and_intent_signals(query_text))

    return ItinerarySynthesizer.union_excluded_tags_from_dietary_keys(key_list)


_DIETARY_CLARIFICATION_QUESTIONS: dict[str, str] = {
    "no_beef": (
        "請問您說不吃牛肉，是指：\n(A) 餐廳菜單完全不能有牛肉\n(B) 您自己不點牛肉，但可以去有牛肉的餐廳"
    ),
    "no_pork": (
        "請問您說不吃豬肉，是指：\n(A) 餐廳菜單完全不能有豬肉\n(B) 您自己不點豬肉，但可以去有豬肉的餐廳"
    ),
    "no_ramen": (
        "請問您說不吃拉麵，是指：\n(A) 餐廳完全不提供或主打拉麵\n(B) 您自己不點拉麵，但可以去有拉麵的店"
    ),
}

_DIETARY_HINT_TO_PREFERS_KEY: dict[str, str] = {
    "no_beef": "prefers_no_beef",
    "no_pork": "prefers_no_pork",
    "no_ramen": "prefers_no_ramen",
}


def _canonical_dietary_hints_list(dh: Any) -> list[str]:
    raw = str(dh or "").strip()
    if not raw:
        return []
    out: list[str] = []
    for part in raw.split(","):
        k = ItinerarySynthesizer._canonical_dietary_constraint_key(part.strip())
        if k:
            out.append(k)
    return list(dict.fromkeys(out))


def _ambiguous_dietary_hint_for_clarification(
    intent: dict[str, Any],
    resolved: dict[str, Any] | None,
) -> str | None:
    dh = intent.get("dietary_hints")
    keys = _canonical_dietary_hints_list(dh)
    if len(keys) != 1:
        return None
    hint_key = keys[0]
    if hint_key not in _DIETARY_CLARIFICATION_QUESTIONS:
        return None
    r = str((resolved or {}).get(hint_key, "") or "").strip().lower()
    if r in {"strict", "loose"}:
        return None
    return hint_key


def _parse_dietary_clarification_reply(query: str) -> str | None:
    q = (query or "").strip().upper()
    ql = (query or "").strip()
    if q in {"A", "(A)", "選A", "Ａ"} or ql.startswith("選項A"):
        return "strict"
    if q in {"B", "(B)", "選B", "Ｂ"} or ql.startswith("選項B"):
        return "loose"
    low = ql.lower()
    if "菜單完全" in ql or "完全不能" in ql or "strict" in low:
        return "strict"
    if "我自己不點" in ql or "不點牛肉" in ql or "不點豬肉" in ql or "不點拉麵" in ql or "loose" in low:
        return "loose"
    return None


def _strip_dietary_hint_key_from_intent(intent: dict[str, Any], hint_key: str) -> None:
    keys = _canonical_dietary_hints_list(intent.get("dietary_hints"))
    if not keys:
        intent["dietary_hints"] = None
        return
    remainder = [x for x in keys if x != hint_key]
    if not remainder:
        intent["dietary_hints"] = None
    else:
        intent["dietary_hints"] = ",".join(remainder)


def _consume_pending_dietary_clarification_answer(state: AgentState) -> bool:
    """If user answered (A)/(B) while a clarification is pending, apply and skip full re-parse."""
    pend = state.get("pending_dietary_clarification")
    if not isinstance(pend, dict) or not pend.get("hint"):
        return False
    mode = _parse_dietary_clarification_reply(str(state.get("query") or ""))
    if mode is None:
        return False
    hint = str(pend["hint"])
    snap = pend.get("intent_snapshot")
    if not isinstance(snap, dict):
        return False
    intent = copy.deepcopy(snap)
    resolved = dict(state.get("dietary_clarification_resolved") or {})
    resolved[hint] = mode
    state["dietary_clarification_resolved"] = resolved
    if mode == "strict":
        pass
    else:
        pref = _DIETARY_HINT_TO_PREFERS_KEY.get(hint)
        _strip_dietary_hint_key_from_intent(intent, hint)
        if pref:
            ec = list(intent.get("explicit_constraints") or [])
            if pref not in ec:
                ec.append(pref)
                intent["explicit_constraints"] = ec
    state["intent"] = intent
    state.pop("pending_dietary_clarification", None)
    state.pop("awaiting_dietary_clarification", None)
    state.pop("clarification_broadcast", None)
    state["plan_excluded_shop_tags"] = sorted(
        _build_plan_excluded_shop_tags(state.get("query") or "", intent, state.get("dietary_profile"))
    )
    state.setdefault("transit_audit", []).append(
        _dj("dietary_clarification_answer", hint=hint, mode=mode)
    )
    return True


class AgentState(TypedDict):
    """LangGraph state; ``intent`` matches ``Intent.as_dict()`` from ``intent_parser``.

    ``intent`` is ``None`` until ``node_route_intent`` runs; each snapshot is a plain dict
    aligned with :meth:`intent_parser.Intent.as_dict` for strict round-trip shape.
    """

    query: str
    research_log: list[str]
    transit_audit: list[str]
    final_itinerary: str
    feedback_updates: list[str]
    learned_weight_profile: dict
    agent_run_id: str
    dietary_profile: dict
    ui_cards: list[dict]
    advanced_mode: bool
    dynamic_shop_pool: list[dict]
    user_locale: str
    wants_flight_search: bool  # kept for intent routing; flight booking removed from main flow
    user_lat: float | None
    user_lng: float | None
    #: Client-supplied LangGraph ``thread_id`` when resuming/checkpoint refinement; empty if none.
    checkpoint_thread_id: str
    #: Prior round finalized itinerary text; filled on thread continuation so intent can treat edits as amendments.
    prev_itinerary: str
    researcher_candidate_names: list[str]
    researcher_notes: str
    auditor_feedback: str
    auditor_rejected: bool
    research_iteration: int
    intent: dict[str, Any] | None  # serialised Intent.as_dict(); None before first parse in a turn
    intent_history: list[dict[str, Any]]  # prior Intent.as_dict() snapshots for correction / audit
    retrieval_history: list[dict]   # append-only; RetrievalReport.as_dict() per round
    critique_history: list[dict]    # append-only; CritiqueReport.as_dict() per round
    synthesis_history: list[dict]   # append-only; SynthesisResult summary per round
    #: Union of lowercase shop tags ruled out via :meth:`decision_engine.ItinerarySynthesizer.DIETARY_EXCLUDED_TAGS`
    #: (intent + dietary profile); set in ``route_intent`` before retriever/plan consume candidates.
    plan_excluded_shop_tags: list[str]
    #: LangGraph checkpoint IDs (one entry per finished user/query round); used for undo / history UX.
    turn_checkpoints: list[str]
    dietary_clarification_resolved: dict[str, str]
    pending_dietary_clarification: dict[str, Any] | None
    awaiting_dietary_clarification: bool
    clarification_broadcast: dict[str, Any] | None
    #: Set by a failing node → downstream nodes noop; orchestrator maps to client errors.
    error: dict[str, Any] | None
    #: Injected deps (``llm_router``, ``openai_chat_client``, ``flight_service``); empty {} uses module defaults.
    runtime_services: dict[str, Any]


class AgentStateModel(BaseModel):
    query: str
    research_log: list[str] = Field(default_factory=list)
    transit_audit: list[str] = Field(default_factory=list)
    final_itinerary: str = ""
    feedback_updates: list[str] = Field(default_factory=list)
    learned_weight_profile: dict = Field(default_factory=dict)
    agent_run_id: str = ""
    dietary_profile: dict = Field(default_factory=lambda: {"ethics": "unspecified", "allergens": [], "religious": "none", "medical": []})
    ui_cards: list[dict] = Field(default_factory=list)
    advanced_mode: bool = False
    dynamic_shop_pool: list[dict] = Field(default_factory=list)
    user_locale: str = ""
    wants_flight_search: bool = False
    user_lat: float | None = None
    user_lng: float | None = None
    checkpoint_thread_id: str = ""
    prev_itinerary: str = ""
    researcher_candidate_names: list[str] = Field(default_factory=list)
    researcher_notes: str = ""
    auditor_feedback: str = ""
    auditor_rejected: bool = False
    research_iteration: int = 0
    intent: dict[str, Any] | None = None
    intent_history: list[dict[str, Any]] = Field(default_factory=list)
    retrieval_history: list[dict] = Field(default_factory=list)
    critique_history: list[dict] = Field(default_factory=list)
    synthesis_history: list[dict] = Field(default_factory=list)
    turn_checkpoints: list[str] = Field(default_factory=list)
    plan_excluded_shop_tags: list[str] = Field(default_factory=list)
    dietary_clarification_resolved: dict[str, str] = Field(default_factory=dict)
    pending_dietary_clarification: dict[str, Any] | None = None
    awaiting_dietary_clarification: bool = False
    clarification_broadcast: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    runtime_services: dict[str, Any] = Field(default_factory=dict)


class AtomicCommitFailure(Exception):
    pass


class OfflineDuffelClient:
    """DEPRECATED: Minimal offline Duffel stub — not in main flow (Task 7 ADR).

    Kept as distributed-tx reference alongside duffel.py / saga.py.
    Full mock implementation is in tests/legacy/test_idempotency.py.
    """

    @staticmethod
    def search_offers(origin: str, destination: str, date: str) -> dict:
        """Return a hardcoded offline offer (TPE-NRT-SFO demo, date ignored)."""
        _ = date
        return {
            "offers": [{"id": f"offline_{origin.lower()}_{destination.lower()}", "total_amount": "199.00", "total_currency": "USD", "slices": [{"segments": [{"origin": {"iata_code": origin}, "destination": {"iata_code": destination}, "operating_carrier": {"iata_code": "OF"}, "operating_carrier_flight_number": "101", "departing_at": "2026-04-24T09:00:00Z", "arriving_at": "2026-04-24T12:00:00Z"}]}]}],
            "passenger_id": "offline_passenger_001",
        }


_SAGA_DIR = Path(os.getenv("SAGA_PERSIST_DIR", str(Path.home() / ".travel_agent" / "saga")))
_SAGA_DIR.mkdir(parents=True, exist_ok=True)
_APP_TZ = ZoneInfo(os.getenv("APP_TIMEZONE", "Asia/Taipei"))
_LEARNING_DB = _SAGA_DIR / "learning_state.db"
# DEPRECATED: kept as distributed-tx reference, not in main flow (see node_flight_search, node_audit)
if os.getenv("OFFLINE_MODE", "0") == "1" or not os.getenv("DUFFEL_ACCESS_TOKEN", "").strip():
    _duffel = OfflineDuffelClient()
else:
    _duffel = DuffelService()
_pref_learner = UserPreferenceLearner()
_llm_router = _LLMRouter()
_learned_weight_profile = WeightProfile.trust_first()
_feedback_samples_seen = 0
_feedback_penalties: dict[str, float] = {}
_taste_max_blacklist: set[str] = set()
_MIN_FEEDBACK_FOR_ML = 200


@dataclass(frozen=True)
class TimeRange:
    start: str | None = None  # HH:MM
    end: str | None = None  # HH:MM


def _init_learning_store() -> None:
    with sqlite3.connect(str(_LEARNING_DB)) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS learner_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                feedback_samples_seen INTEGER NOT NULL,
                trust_bias REAL NOT NULL,
                preference_bias REAL NOT NULL,
                logistics_bias REAL NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback_penalties (
                shop_name TEXT PRIMARY KEY,
                penalty REAL NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT OR IGNORE INTO learner_state
            (id, feedback_samples_seen, trust_bias, preference_bias, logistics_bias)
            VALUES (1, 0, 0.8, 0.1, 0.1)
            """
        )
        conn.commit()


def _load_learning_state() -> None:
    global _feedback_samples_seen, _learned_weight_profile, _feedback_penalties
    _init_learning_store()
    with sqlite3.connect(str(_LEARNING_DB)) as conn:
        row = conn.execute(
            """
            SELECT feedback_samples_seen, trust_bias, preference_bias, logistics_bias
            FROM learner_state WHERE id = 1
            """
        ).fetchone()
    if row is None:
        return
    _feedback_samples_seen = int(row[0])
    _learned_weight_profile = ScoringEngine.normalize_weights(
        WeightProfile(trust_bias=float(row[1]), preference_bias=float(row[2]), logistics_bias=float(row[3]))
    )
    with sqlite3.connect(str(_LEARNING_DB)) as conn:
        rows = conn.execute("SELECT shop_name, penalty FROM feedback_penalties").fetchall()
    _feedback_penalties = {str(name): float(penalty) for name, penalty in rows}


def _save_learning_state() -> None:
    with sqlite3.connect(str(_LEARNING_DB)) as conn:
        conn.execute(
            """
            UPDATE learner_state
            SET feedback_samples_seen = ?, trust_bias = ?, preference_bias = ?, logistics_bias = ?
            WHERE id = 1
            """,
            (
                int(_feedback_samples_seen),
                float(_learned_weight_profile.trust_bias),
                float(_learned_weight_profile.preference_bias),
                float(_learned_weight_profile.logistics_bias),
            ),
        )
        conn.execute("DELETE FROM feedback_penalties")
        conn.executemany(
            "INSERT INTO feedback_penalties(shop_name, penalty) VALUES(?, ?)",
            [(name, float(p)) for name, p in _feedback_penalties.items() if float(p) > 1e-9],
        )
        conn.commit()


def make_initial_state(
    query: str,
    dietary_profile: dict | None = None,
    agent_run_id: str | None = None,
    advanced_mode: bool = False,
    user_locale: str | None = None,
    user_lat: float | None = None,
    user_lng: float | None = None,
    *,
    checkpoint_thread_id: str | None = None,
    prev_itinerary: str = "",
) -> dict:
    run_id = agent_run_id or uuid.uuid4().hex
    ctid = (checkpoint_thread_id or "").strip()
    model = AgentStateModel(
        query=query,
        dietary_profile=dietary_profile or {"ethics": "unspecified", "allergens": [], "religious": "none", "medical": []},
        agent_run_id=run_id,
        advanced_mode=advanced_mode,
        user_locale=(user_locale or "").strip(),
        user_lat=user_lat,
        user_lng=user_lng,
        checkpoint_thread_id=ctid,
        prev_itinerary=(prev_itinerary or ""),
    )
    return model.model_dump(mode="json")


def _cleanup_old_txn_logs(retention_days: int = 30) -> None:
    cutoff = datetime.now(timezone.utc).timestamp() - retention_days * 86400
    for p in _SAGA_DIR.glob("saga_txn_*.json"):
        try:
            if p.stat().st_mtime < cutoff:
                p.unlink(missing_ok=True)
        except Exception:
            continue


_load_learning_state()


logger = logging.getLogger(__name__)
_default_openai_chat_client = OpenAIChatCompletionClient()


def _runtime_services_map(state: AgentState | dict[str, Any]) -> dict[str, Any]:
    raw = state.get("runtime_services") if hasattr(state, "get") else None
    if isinstance(raw, dict):
        return raw
    return {}


def _svc_llm_router(state: AgentState):
    svc = _runtime_services_map(state)
    lr = svc.get("llm_router")
    return lr if lr is not None else _llm_router


def _svc_openai_chat_client(state: AgentState) -> OpenAIChatCompletionClient:
    svc = _runtime_services_map(state)
    c = svc.get("openai_chat_client")
    if isinstance(c, OpenAIChatCompletionClient):
        return c
    return _default_openai_chat_client


def _svc_flight_service(state: AgentState):
    """Duffel-compatible flight Offers API adapter (offline stub or ``DuffelService``)."""
    svc = _runtime_services_map(state)
    fs = svc.get("flight_service")
    if fs is None and "duffel" in svc:
        fs = svc.get("duffel")
    return fs if fs is not None else _duffel


def _attach_node_error(
    state: AgentState,
    node: str,
    exc_or_message: BaseException | str,
    *,
    code: str = "NODE_FAILURE",
) -> None:
    if state.get("error"):
        logger.warning(
            "agent error already set (%s); not overwriting from node=%s",
            (state.get("error") or {}).get("node"),
            node,
        )
        return
    if isinstance(exc_or_message, BaseException):
        msg = str(exc_or_message)
        detail = "".join(traceback.format_exception(exc_or_message)).strip()
        logger.exception("Agent node %s captured exception", node)
    else:
        msg = str(exc_or_message)
        detail = ""
    state["error"] = {
        "node": node,
        "message": msg,
        "code": code,
        "detail": detail[:8000],
    }


def _finalize_plan_on_agent_error(state: AgentState) -> AgentState:
    err = state.get("error") or {}
    nm = str(err.get("node", "?"))
    msg = str(err.get("message", "unknown_error"))
    state["final_itinerary"] = f"## 行程無法生成\n\n- **發生於** `{nm}`\n- **原因** {msg}\n"
    state.setdefault("ui_cards", list(state.get("ui_cards") or []))
    state.setdefault("transit_audit", []).append(_dj("plan_aborted_agent_error", node=nm, message=msg))
    return state


def _is_right_now_mode(query: str) -> bool:
    q = (query or "").upper()
    return ("RIGHT_NOW" in q) or ("現在餓了" in query) or ("附近 30 分鐘" in query)


def _is_taipei_mode(query: str) -> bool:
    q = (query or "").upper()
    return ("台北" in query) or ("TAIPEI" in q) or ("台灣" in query) or ("TAIWAN" in q)


def _is_tokyo_mode(query: str) -> bool:
    q = (query or "").upper()
    return ("東京" in query) or ("TOKYO" in q)


def _region_driven_mock_fixtures(city: str, region: str) -> tuple[dict[str, str], dict[tuple[str, str], str]]:
    """
    Region-aware defaults for SNS/traffic adapters.
    Keep deterministic fixtures for offline QA runs.
    """
    city_norm = (city or "").strip().lower()
    if region == "tw":
        return (
            {
                "lindongfang_beef": "本日正常營業",
                "fuhang_official": "今日照常營業",
                "moonmoonfood": "正常營業",
            },
            {
                ("板橋", "東區"): "車流偏高，預估延誤 18 分鐘",
                ("東區", "西門町"): "",
            },
        )
    if region == "jp" and ("東京" in city or "tokyo" in city_norm):
        return (
            {
                "fuunji_shinjuku": "通常営業",
                "tsujihan_nihonbashi": "本日通常営業",
            },
            {
                ("新宿", "渋谷"): "山手線混雑",
                ("渋谷", "日本橋"): "",
            },
        )
    return (
        {
            "moeyo_mensuke": "通常営業",
            "harbs_official": "本日通常営業",
        },
        {
            ("Kyoto Station", "Umeda"): "大幅延誤",
            ("Umeda", "Karasuma"): "",
        },
    )


def _query_implied_dietary_ethics(query: str) -> str | None:
    q = (query or "").lower()
    if "vegan" in q or "純素" in q:
        return "vegan"
    if "vegetarian" in q or "素食" in q:
        return "vegetarian"
    if "pescatarian" in q:
        return "pescatarian"
    return None


def _requested_meal_count(query: str) -> int | None:
    q = (query or "").lower()
    patterns = [
        r"([1-5])\s*餐",
        r"([1-5])\s*meals?",
        r"(一|二|三|四|五)\s*餐",
        r"(一|二|三|四|五)\s*[頓顿]",
    ]
    cn_map = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5}
    for pat in patterns:
        m = re.search(pat, q)
        if not m:
            continue
        token = m.group(1)
        if token.isdigit():
            return max(1, min(5, int(token)))
        if token in cn_map:
            return cn_map[token]
    return None


def _requested_meal_slots(query: str) -> list[str]:
    """
    Infer which named meal slots appear in the query / count expansion.
    Per-slot tag OR-groups (tea/dinner, etc.) are enforced separately via
    `_slot_level_required_tags`; global intent tags use `explicit_category_tags`.
    """
    q = (query or "").lower()
    requested_count = _requested_meal_count(query)
    slots: list[str] = []
    rules: list[tuple[str, tuple[str, ...]]] = [
        ("breakfast", ("早餐", "早午餐", "morning", "breakfast", "brunch")),
        ("lunch", ("午餐", "中餐", "lunch", "noon")),
        ("tea", ("下午茶", "茶點", "tea", "cafe", "coffee break")),
        ("dinner", ("晚餐", "晚飯", "dinner", "supper")),
        ("late_night", ("消夜", "宵夜", "late night", "late_night", "midnight snack")),
    ]
    for slot, keys in rules:
        if any(k in q for k in keys):
            slots.append(slot)
    tr = _extract_user_time_window(query)
    forced_start_hhmm = tr.start
    morning_start = False
    if forced_start_hhmm:
        hh, mm = [int(x) for x in forced_start_hhmm.split(":", 1)]
        morning_start = (hh, mm) <= (10, 30)
    if morning_start and "breakfast" not in slots:
        slots = ["breakfast", *slots]

    # De-duplicate while preserving order.
    deduped_slots: list[str] = []
    seen_slots: set[str] = set()
    for s in slots:
        if s not in seen_slots:
            deduped_slots.append(s)
            seen_slots.add(s)
    slots = deduped_slots

    if requested_count is not None:
        target = max(1, min(5, requested_count))
        morning_hint = morning_start or any(k in q for k in ("早上", "清晨", "早餐", "morning", "breakfast"))
        ramen_hint = _is_ramen_intent(query)
        if target == 3 and ramen_hint and morning_hint:
            return ["breakfast", "lunch", "dinner"]

        default_order = (
            ["breakfast", "lunch", "tea", "dinner", "late_night"]
            if morning_hint
            else ["lunch", "tea", "dinner", "breakfast", "late_night"]
        )
        for slot in default_order:
            if len(slots) >= target:
                break
            if slot not in slots:
                slots.append(slot)
        return slots[:target]
    return slots


def _effective_plan_meal_slots(intent: dict, query: str) -> list[str]:
    """Prefer explicit planner intent; otherwise derive from `_requested_meal_slots` query expansion."""
    raw = intent.get("meal_slots") if isinstance(intent.get("meal_slots"), list) else None
    if isinstance(raw, list) and raw:
        base = list(raw)
    else:
        base = _requested_meal_slots(query)
    return ItinerarySynthesizer._normalize_slot_sequence(base)


def _inject_slot_anchor_rankeds_into_phase1(
    *,
    core_head: list[RankedShop],
    full_scores: list[RankedShop],
    seed_profiles: list[ShopProfile],
    slots: list[str],
    query: str,
    cap: int,
) -> list[RankedShop]:
    """Prepend researcher-style seed anchors per slot before DP so each layer has feasible nodes."""
    norm_slots = ItinerarySynthesizer._normalize_slot_sequence(slots)
    if not norm_slots:
        return core_head[:cap]
    anchor_names = _researcher_slot_anchor_names(norm_slots, seed_profiles, query=query)
    by_name = {r.shop.name: r for r in full_scores}
    seen: set[str] = set()
    head: list[RankedShop] = []
    for nm in anchor_names:
        rs = by_name.get(nm)
        if rs is None:
            continue
        if nm in seen:
            continue
        seen.add(nm)
        head.append(rs)
    merged: list[RankedShop] = list(head)
    for rs in core_head:
        if rs.shop.name not in seen:
            seen.add(rs.shop.name)
            merged.append(rs)
        if len(merged) >= cap:
            break
    return merged[:cap]


def _normalize_hhmm_from_period(hour: int, minute: int, period: str | None) -> str:
    h = max(0, min(23, int(hour)))
    m = max(0, min(59, int(minute)))
    if period:
        p = period.lower()
        if p in {"下午", "晚上", "pm", "p.m.", "傍晚"} and h < 12:
            h += 12
        if p in {"凌晨"} and h == 12:
            h = 0
    return f"{h:02d}:{m:02d}"


def _extract_user_time_window(query: str) -> TimeRange:
    text = query or ""
    m = re.search(r"(\d{1,2}:\d{2})\s*[-–~～到至]\s*(\d{1,2}:\d{2})", text)
    if m:
        start = _clamp_hhmm_token(m.group(1))
        end = _clamp_hhmm_token(m.group(2))
        return TimeRange(start=start, end=end)

    # e.g. "早上7點到晚上9點", "7點到21點"
    m2 = re.search(
        r"(早上|上午|中午|下午|晚上|凌晨|pm|am)?\s*(\d{1,2})(?::(\d{2}))?\s*點?\s*[-–~～到至]\s*"
        r"(早上|上午|中午|下午|晚上|凌晨|pm|am)?\s*(\d{1,2})(?::(\d{2}))?\s*點?",
        text,
        flags=re.IGNORECASE,
    )
    if m2:
        p1, h1, m1, p2, h2, m2v = m2.group(1), m2.group(2), m2.group(3), m2.group(4), m2.group(5), m2.group(6)
        start = _normalize_hhmm_from_period(int(h1), int(m1 or "0"), p1)
        end = _normalize_hhmm_from_period(int(h2), int(m2v or "0"), p2)
        return TimeRange(start=start, end=end)

    # e.g. "7點開始", "早上7点开始"; used by meal-slot expansion so 7 AM plans open with breakfast anchors.
    m3 = re.search(
        r"(早上|清晨|凌晨|上午|中午|下午|晚上)?\s*(\d{1,2})\s*[點:]?\s*(\d{2})?\s*(開始|开始)",
        text,
        flags=re.IGNORECASE,
    )
    if m3:
        period, h_raw, mm_raw = m3.group(1), m3.group(2), m3.group(3)
        start = _normalize_hhmm_from_period(int(h_raw), int(mm_raw or "0"), period)
        return TimeRange(start=start, end=None)

    return TimeRange()


def _extract_time_window(query: str) -> tuple[str | None, str | None]:
    # Backward-compatible alias.
    tr = _extract_user_time_window(query)
    return tr.start, tr.end


def _is_ramen_intent(query: str) -> bool:
    q = (query or "").lower()
    return any(k in q for k in ("拉麵", "拉面", "ramen"))


def _has_strong_ramen_intent(query: str) -> bool:
    # Strong intent: ramen keyword + explicit meal-planning context (e.g. "三餐拉麵").
    if not _is_ramen_intent(query):
        return False
    requested_n = _requested_meal_count(query)
    if requested_n is not None:
        return True
    q = (query or "").lower()
    return any(k in q for k in ("行程", "schedule", "itinerary", "只吃", "全都"))


def _extract_explicit_category_tags(query: str) -> set[str]:
    q_raw = query or ""
    q = q_raw.lower()
    tags: set[str] = set()
    if (any(k in q for k in ("拉麵", "拉面", "ramen"))) and not _user_negates_food_category_in_query(q_raw, "ramen"):
        tags.add("ramen")
    if (any(k in q for k in ("居酒屋", "izakaya"))) and not _user_negates_food_category_in_query(q_raw, "izakaya"):
        tags.add("izakaya")
    kws_l = [k.lower() for k in _extract_search_category_keywords(query)]
    if (
        any(x in kws_l for x in ("cake", "bakery", "dessert", "patisserie", "coffee", "cafe"))
        and not _user_negates_food_category_in_query(q_raw, "dessert")
    ):
        tags.add("dessert")
    if any(x in kws_l for x in ("yakitori", "izakaya", "beer", "pub", "japanese pub")):
        if not _user_negates_food_category_in_query(q_raw, "izakaya"):
            tags.add("izakaya")
    if "ramen" in kws_l and not _user_negates_food_category_in_query(q_raw, "ramen"):
        tags.add("ramen")
    return tags


def _direct_food_category_mentions(query: str) -> set[str]:
    """使用者在本輪 query 中明確說出口的食物類別（不含搜尋擴展／歷史偏好）。"""
    q_raw = query or ""
    q = q_raw.lower()
    out: set[str] = set()
    if (
        any(k in q_raw for k in ("拉麵", "拉面")) or "ramen" in q
    ) and not _user_negates_food_category_in_query(q_raw, "ramen"):
        out.add("ramen")
    if (
        any(
            k in q_raw
            for k in ("蛋糕", "甜點", "甜点", "下午茶", "茶點", "巴斯克", "提拉米蘇", "抹茶", "戚風")
        )
        and not _user_negates_food_category_in_query(q_raw, "dessert")
    ):
        out.add("dessert")
    if (
        any(k in q_raw for k in ("串燒", "串烧", "燒鳥", "烧鸟", "焼き鳥")) or "yakitori" in q
    ) and not _user_negates_food_category_in_query(q_raw, "izakaya"):
        out.add("izakaya")
    if (
        any(k in q_raw for k in ("居酒屋", "啤酒", "生啤")) or "izakaya" in q
    ) and not _user_negates_food_category_in_query(q_raw, "izakaya"):
        out.add("izakaya")
    return out


def _softened_explicit_category_tags(query: str) -> set[str]:
    """
    保留顯式檢索標籤，但若使用者明確轉向甜點／串燒／居酒屋且未說拉麵，
    則移除僅由關鍵字擴展帶入的 ramen，避免與當輪意圖衝突。
    """
    tags = _extract_explicit_category_tags(query)
    direct = _direct_food_category_mentions(query)
    slot_req = _slot_level_required_tags(query)
    out = set(tags)
    if "ramen" not in direct and "ramen" in out:
        if direct & {"dessert", "izakaya"} or slot_req:
            out.discard("ramen")
    return out


def _plan_global_explicit_tags(query: str) -> set[str]:
    """
    全域 HARD_TAG／DP 覆蓋標籤：當時段標籤已涵蓋本輪甜點／居酒屋意圖時，
    不再用單一全域集合對整個行程施加强制（改由 slot_required_tags 篩選）。
    """
    softened = _softened_explicit_category_tags(query)
    slot_req = _slot_level_required_tags(query)
    if not slot_req:
        return softened
    slot_union: set[str] = set()
    for v in slot_req.values():
        slot_union |= v
    direct = _direct_food_category_mentions(query)
    # 若僅剩擴展標籤且已全部落在 slot OR 集合內，且使用者未明示拉麵 → 不做全域硬過濾
    if softened and softened <= slot_union and "ramen" not in direct:
        return set()
    # 否則仍保留「語意不在 slot_union」的標籤（例如僅午餐要拉麵）
    return {t for t in softened if t not in slot_union or t in direct}


def _should_damp_preference_for_query(query: str) -> bool:
    """當輪query 明示甜點／串燒／居酒屋等，或已有時段標籤約束時，應削弱歷史口味權重。"""
    if _slot_level_required_tags(query):
        return True
    direct = _direct_food_category_mentions(query)
    return bool(direct & {"dessert", "izakaya"})


def _apply_runtime_weight_damping(wp: WeightProfile) -> WeightProfile:
    """略提高 trust、壓低 preference_bias，讓顯式搜尋／標籤優先於長期偏好。"""
    return ScoringEngine.normalize_weights(
        WeightProfile(
            trust_bias=min(1.0, wp.trust_bias + 0.12),
            preference_bias=max(0.05, wp.preference_bias * 0.52),
            logistics_bias=wp.logistics_bias,
        )
    )


def _slot_level_required_tags(query: str) -> dict[str, set[str]]:
    """
    時段專用標籤約束（與全域 explicit_category_tags 聯動；各 slot 為 OR）。
    例：下午茶 + 甜點意圖 → tea 時段須命中 cake／dessert／cafe…；
    晚餐 + 串燒／啤酒／居酒屋 → dinner 須命中 yakitori／izikaya…。
    """
    slots_plan = _requested_meal_slots(query)
    slots_set = set(slots_plan)
    q_raw = query or ""
    q = q_raw.lower()
    search_l = [k.lower() for k in _extract_search_category_keywords(query)]
    dessert_kw = frozenset({"cake", "bakery", "dessert", "patisserie", "coffee", "cafe"})
    izakaya_kw = frozenset({"yakitori", "izakaya", "beer", "pub", "japanese pub", "grilled chicken"})
    out: dict[str, set[str]] = {}

    wants_tea_dessert = (
        any(k in q_raw for k in ("蛋糕", "甜點", "甜点", "下午茶", "巴斯克", "提拉米蘇", "抹茶"))
        or ("下午茶" in q_raw)
        or ("afternoon tea" in q)
        or bool(dessert_kw & set(search_l))
    )
    if "tea" in slots_set and wants_tea_dessert:
        out["tea"] = {
            "cake",
            "dessert",
            "bakery",
            "patisserie",
            "cafe",
            "afternoon_tea",
            "tea",
            "refresh",
        }

    wants_dinner_iza = (
        any(
            k in q_raw
            for k in ("串燒", "串烧", "燒鳥", "烧鸟", "啤酒", "生啤", "居酒屋", "焼き鳥", "yakitori", "izakaya")
        )
        or bool(izakaya_kw & set(search_l))
    )
    if "dinner" in slots_set and wants_dinner_iza:
        out["dinner"] = {"yakitori", "izakaya", "nightlife", "beer"}

    return out


# Minimum tag coverage from SEED pool per meal slot (any intersection => slot covered).
_SLOT_TAG_COVERAGE: dict[str, frozenset[str]] = {
    "breakfast": frozenset({"breakfast", "brunch", "morning", "ramen", "cafe"}),
    "lunch": frozenset({"lunch", "main_meal", "ramen", "quick_meal"}),
    "tea": frozenset({"tea", "dessert", "cafe", "cake", "bakery", "afternoon_tea", "coffee", "refresh"}),
    "dinner": frozenset({"dinner", "main_meal", "ramen", "course", "izakaya", "kaiseki", "sukiyaki"}),
    "late_night": frozenset({"late_night", "izakaya", "ramen", "nightlife", "night_food", "late_open"}),
}

# Slot-triggered English bundles (Places text search): one independent query per meal slot in plan.
_SLOT_FORCED_PLACES_QUERY: dict[str, str] = {
    "breakfast": "breakfast brunch coffee morning cafe egg sandwich",
    "lunch": "lunch restaurant bento noodles quick meal",
    # Afternoon tea / cake visibility (HYBRID_PHASE1 dessert pool)
    "tea": "dessert shop cake bakery patisserie afternoon tea cafe coffee sweets confectionery",
    # Dinner yakitori / izakaya visibility
    "dinner": "izakaya yakitori skewer kushiyaki japanese pub grill dinner beer",
    "late_night": "izakaya ramen late night bar yakitori night snack",
}


def _extract_search_category_keywords(query: str) -> list[str]:
    """
    Map multi-type user phrases to English tokens for Google Places text search.
    Order preserved; duplicates skipped.
    """
    q = (query or "").lower()
    out: list[str] = []

    def add(*words: str) -> None:
        for w in words:
            wl = w.lower()
            if wl not in [x.lower() for x in out]:
                out.append(w)

    if any(k in q for k in ("蛋糕", "戚風", "提拉米蘇", "巴斯克")):
        add("cake", "bakery", "dessert")
    if any(k in q for k in ("甜點", "甜点", "下午茶", "茶點", "點心", "点心")):
        add("dessert", "patisserie", "cafe")
    if any(k in q for k in ("咖啡", "珈琲", "手沖", "拿鐵", "美式")):
        add("coffee", "cafe")
    if any(k in q for k in ("串燒", "串烧", "燒鳥", "烧鸟", "焼き鳥", "yakitori")):
        add("yakitori", "grilled chicken")
    if any(k in q for k in ("啤酒", "生啤", "draft beer", "黑啤")):
        add("beer", "pub")
    if any(k in q for k in ("居酒屋", "izakaya")):
        add("izakaya", "japanese pub")
    if _is_ramen_intent(query):
        add("ramen", "noodles")
    if any(k in q for k in ("壽司", "寿司", "sushi")):
        add("sushi")
    if any(k in q for k in ("燒肉", "烧肉", "烤肉", "yakiniku")):
        add("yakiniku", "bbq")
    return out


def _seed_shop_tag_bag(shop: ShopProfile) -> set[str]:
    return {str(t).lower() for t in shop.tags} | {str(t).lower() for t in shop.occasion_tags}


def _seed_single_covers_slot(shop: ShopProfile, slot: str) -> bool:
    """True if one shop semantically overlaps _SLOT_TAG_COVERAGE[slot]."""
    needed = _SLOT_TAG_COVERAGE.get(slot)
    if not needed:
        return True
    return bool(_seed_shop_tag_bag(shop) & needed)


def _seed_covers_meal_slot(seed_shops: list[ShopProfile], slot: str) -> bool:
    """True if at least one seed shop has tags/occasion overlap with slot needs."""
    return any(_seed_single_covers_slot(s, slot) for s in seed_shops)


def _researcher_semantic_overlap(shop: ShopProfile, slot: str) -> bool:
    """Broad slot semantics for researcher coverage (tea/late-night include common snack tokens)."""
    base = frozenset(_SLOT_TAG_COVERAGE.get(slot, frozenset()))
    if slot == "tea":
        base |= frozenset({"snack", "tangyuan", "tang_yuan"})
    elif slot == "late_night":
        base |= frozenset({"snack", "night_snack"})
    if not base:
        return True
    return bool(_seed_shop_tag_bag(shop) & base)


def _researcher_clock_minutes(hhmm: str | None) -> int | None:
    """Parse HH:MM to minute-of-day; None on failure."""
    try:
        parts = str(hhmm or "").strip().split(":", 1)
        hh = max(0, min(23, int(parts[0])))
        mm = max(0, min(59, int(parts[1]) if len(parts) > 1 else 0))
        return hh * 60 + mm
    except Exception:
        return None


def _researcher_slot_required_gate(shop: ShopProfile, slot: str, slot_req: dict[str, set[str]]) -> bool:
    """Each slot maps to optional OR-groups from `_slot_level_required_tags`."""
    if not slot_req:
        return True
    need = slot_req.get(str(slot).lower())
    if not need:
        return True
    return bool(_seed_shop_tag_bag(shop) & need)


def _researcher_time_fit_slot(shop: ShopProfile, slot: str, tier: str) -> bool:
    """
    Lightweight open/close heuristics aligned with itinerary meal windows.
    tier: strict (time + semantics), relaxed (wider clocks), semantic_only skips time.
    """
    if tier == "semantic_only":
        return True
    open_m = _researcher_clock_minutes(getattr(shop, "open_time", "") or "")
    close_m = _researcher_clock_minutes(getattr(shop, "close_time", "") or "")
    bag = _seed_shop_tag_bag(shop)

    # Breakfast: early starters or breakfast-tagged brunch places opening by late morning.
    if slot == "breakfast":
        if not _researcher_semantic_overlap(shop, "breakfast"):
            return False
        if tier == "relaxed":
            if open_m is None:
                return True
            if open_m < 8 * 60:
                return True
            brunchish = bag & {"breakfast", "brunch", "morning", "soy_milk", "coffee", "cafe"}
            return bool(brunchish) and open_m <= (11 * 60 + 30)
        # strict
        if open_m is None:
            return True
        if open_m < 8 * 60:
            return True
        brunchish = bag & {"breakfast", "brunch", "morning", "soy_milk"}
        return bool(brunchish) and open_m <= (10 * 60 + 45)

    if slot == "lunch":
        if not _researcher_semantic_overlap(shop, "lunch"):
            return False
        if open_m is None:
            return True
        if tier == "relaxed":
            return (10 * 60 + 30) <= open_m <= (13 * 60 + 30)
        return (11 * 60) <= open_m <= (12 * 60 + 59)

    if slot == "tea":
        return _researcher_semantic_overlap(shop, "tea")

    if slot == "dinner":
        if not _researcher_semantic_overlap(shop, "dinner"):
            return False
        if close_m is None:
            return True
        if tier == "relaxed":
            return close_m >= (20 * 60 + 30)
        return close_m >= (21 * 60)

    if slot == "late_night":
        if _researcher_semantic_overlap(shop, "late_night"):
            if close_m is None:
                return True
            if tier == "relaxed":
                return close_m >= (22 * 60 + 30) or close_m <= (9 * 60)
            return close_m >= (23 * 60) or close_m <= (9 * 60)
        if close_m is None:
            return False
        return close_m >= (23 * 60) if tier == "strict" else close_m >= (22 * 60 + 30)

    return _researcher_semantic_overlap(shop, slot)


def _researcher_seed_eligible_for_slot(
    shop: ShopProfile,
    slot: str,
    *,
    query: str,
    tier: str,
) -> bool:
    if not _researcher_slot_required_gate(shop, slot, _slot_level_required_tags(query)):
        return False
    if tier == "semantic_only":
        return _researcher_semantic_overlap(shop, slot)
    sem_ok = _researcher_semantic_overlap(shop, slot)
    if not sem_ok:
        return False
    return _researcher_time_fit_slot(shop, slot, tier=tier)


def _researcher_shop_eligible_any_tier(shop: ShopProfile, slot: str, *, query: str) -> bool:
    return any(
        _researcher_seed_eligible_for_slot(shop, slot, query=query, tier=t)
        for t in ("strict", "relaxed", "semantic_only")
    )


def _researcher_score_tuple(shop: ShopProfile) -> tuple[float, float, float]:
    ss = getattr(shop, "source_scores", None) or {}
    table_top = float(max(ss.values()) if ss else 0.0)
    return (
        float(shop.google_rating or 0.0),
        table_top,
        float(shop.trust_score),
    )


def _researcher_best_seed_for_slot(
    seed_shops: list[ShopProfile],
    slot: str,
    *,
    query: str,
    forbidden_names: set[str],
) -> ShopProfile | None:
    """Pick highest-scoring seed shop for slot; relax time tiers if sparse."""
    for tier in ("strict", "relaxed", "semantic_only"):
        pool = [
            s
            for s in seed_shops
            if s.name not in forbidden_names
            and _researcher_seed_eligible_for_slot(s, slot, query=query, tier=tier)
        ]
        if not pool:
            continue
        pool.sort(key=_researcher_score_tuple, reverse=True)
        return pool[0]
    return None


def _researcher_slot_anchor_names(
    meal_slots: list[str],
    seed_shops: list[ShopProfile],
    *,
    query: str,
) -> list[str]:
    """At most one anchored seed candidate per meal slot (names unique when possible)."""
    out: list[str] = []
    seen: set[str] = set()
    for slot in meal_slots:
        pick = _researcher_best_seed_for_slot(seed_shops, slot, query=query, forbidden_names=seen)
        if pick is None:
            continue
        out.append(pick.name)
        seen.add(pick.name)
    return out


def _researcher_finalize_candidate_names(
    *,
    meal_slots: list[str],
    shops: list[ShopProfile],
    seed_shops: list[ShopProfile],
    query: str,
    base_names: list[str],
    auditor_feedback: str = "",
    notes_suffix: str = "",
) -> tuple[list[str], str]:
    """
    Prepend anchored per-slot picks from seed catalog, preserve order uniqueness,
    then append other high-score shops until cap.
    """
    shop_by_name = {s.name: s for s in shops}
    anchors = _researcher_slot_anchor_names(meal_slots, seed_shops, query=query)
    seen: set[str] = set()
    merged: list[str] = []

    def push(name: str) -> None:
        if name not in shop_by_name or name in seen:
            return
        seen.add(name)
        merged.append(name)

    for a in anchors:
        push(a)
    for n in base_names:
        push(str(n).strip())

    floor = len(meal_slots) if meal_slots else 4
    cap = max(8, floor + 3, len(anchors) + 4)

    q = (query or "").lower()
    avoid_far = ("太遠" in auditor_feedback) or ("distance" in auditor_feedback.lower())
    sorted_shops = sorted(shops, key=_researcher_score_tuple, reverse=True)
    for s in sorted_shops:
        if len(merged) >= cap:
            break
        tags_low = {str(t).lower() for t in s.tags}
        if "ramen" in q and "ramen" not in tags_low and "noodle" not in tags_low:
            continue
        if "vegan" in q and "vegan" not in [x.lower() for x in s.allowed_dietary_preferences]:
            continue
        if avoid_far and (s.latitude is None or s.longitude is None):
            continue
        push(s.name)

    notes = notes_suffix.strip()
    anchor_note = ""
    if anchors:
        anchor_note = f"slot_anchors={'|'.join(anchors)};"
    extras = "; ".join(x for x in (anchor_note, notes) if x)
    return merged, extras


def _plan_dynamic_place_queries(query: str, city: str, seed_shops: list[ShopProfile]) -> tuple[list[str], list[str]]:
    """
    Returns (ordered_unique_queries, uncovered_slots).

    Slot-triggered search: for every slot from `_requested_meal_slots`, append a dedicated
    Places query (`_SLOT_FORCED_PLACES_QUERY`) **before** the raw user query so HYBRID_PHASE1
    receives dessert / izakaya / … candidates even when the seed catalog already covers tags.
    """
    slots = _requested_meal_slots(query)
    uncovered = [sl for sl in slots if not _seed_covers_meal_slot(seed_shops, sl)]
    intent_kw = _extract_search_category_keywords(query)

    planned: list[str] = []

    # 1) Per-slot retrieval (tea → dessert shop bundle, dinner → izakaya/yakitori, …)
    for sl in slots:
        bundle = _SLOT_FORCED_PLACES_QUERY.get(sl)
        if bundle:
            planned.append(f"{bundle} {city}")

    # 2) Full user wording (semantic + entities)
    base = (query or "").strip()
    if base:
        planned.append(base)

    # 3) Keyword-expanded composite line
    if intent_kw:
        planned.append(f"{' '.join(intent_kw)} {city}")

    seen: set[str] = set()
    unique: list[str] = []
    for p in planned:
        key = " ".join(p.split()).lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(p.strip())
    return unique, uncovered


def _is_appetite_light_intent(query: str) -> bool:
    q = query or ""
    return any(k in q for k in ("吃不太下", "吃不下", "食量小", "小食量", "胃口小"))


def _feedback_complains_fame_unreliable(query: str) -> bool:
    """使用者抱怨名氣／獎項不值得信賴時，改為偏好優先權重。"""
    q = query or ""
    low = q.lower()
    if "feedback:" not in low:
        return False
    tail = low.split("feedback:", 1)[-1]
    if "名氣不準" in tail or "名气不准" in tail:
        return True
    if "畢比登很雷" in tail or "毕比登很雷" in tail:
        return True
    return False


def _locale_token_to_city_region(token: str) -> tuple[str, str] | None:
    """Map a locale hint (user profile or DEFAULT_LOCALE_CITY) to (city, region)."""
    raw = (token or "").strip()
    if not raw:
        return None
    q = raw.lower()
    if ("台北" in raw) or ("台灣" in raw) or ("taipei" in q) or ("taiwan" in q):
        return ("台北", "tw")
    if ("京都" in raw) or ("kyoto" in q):
        return ("京都", "jp")
    if ("東京" in raw) or ("tokyo" in q):
        return ("東京", "jp")
    if ("大阪" in raw) or ("osaka" in q):
        return ("大阪", "jp")
    return None


def _infer_city_region_from_coords(user_lat: float | None, user_lng: float | None) -> tuple[str, str] | None:
    """Very light reverse-geo buckets for default region routing."""
    if user_lat is None or user_lng is None:
        return None
    lat = float(user_lat)
    lng = float(user_lng)
    # Taipei metro rough bbox
    if 24.8 <= lat <= 25.3 and 121.3 <= lng <= 121.8:
        return ("台北", "tw")
    # Tokyo metro rough bbox
    if 35.4 <= lat <= 36.0 and 139.4 <= lng <= 140.1:
        return ("東京", "jp")
    return None


def _extract_city_from_query(
    query: str,
    user_locale: str | None = None,
    user_lat: float | None = None,
    user_lng: float | None = None,
) -> tuple[str, str]:
    """
    Resolve default city for Places / expansion.
    Priority: explicit query intent > user_locale > user coordinates > DEFAULT_LOCALE_CITY > Kyoto.
    """
    q_lower = (query or "").lower()
    q_raw = query or ""
    if ("台北" in q_raw) or ("taipei" in q_lower) or ("台灣" in q_raw) or ("taiwan" in q_lower):
        return ("台北", "tw")
    if ("京都" in q_raw) or ("kyoto" in q_lower):
        return ("京都", "jp")
    if ("東京" in q_raw) or ("tokyo" in q_lower):
        return ("東京", "jp")
    if ("大阪" in q_raw) or ("osaka" in q_lower):
        return ("大阪", "jp")

    ul = _locale_token_to_city_region(user_locale or "")
    if ul is not None:
        return ul

    coord_guess = _infer_city_region_from_coords(user_lat=user_lat, user_lng=user_lng)
    if coord_guess is not None:
        return coord_guess

    env_hint = os.getenv("DEFAULT_LOCALE_CITY", "").strip()
    env_mapped = _locale_token_to_city_region(env_hint)
    if env_mapped is not None:
        return env_mapped

    return ("京都", "jp")


def _is_flight_booking_intent(query: str) -> bool:
    """Route to Duffel flight search when query looks like a flight booking request."""
    q = query or ""
    low = q.lower()
    if "機票" in q:
        return True
    if "flight" in low:
        return True
    if "book" in low:
        return True
    return False


# Dynamic pool from Places: below this count, node_plan retries with broader geo queries (+ optional LLM).
_MIN_DYNAMIC_POOL_HEALTHY = 5


def _fallback_broad_geo_queries(query: str, city: str, region: str) -> list[str]:
    """Deterministic wider-area queries when LLM is unavailable or fails."""
    locality = (city or "").strip() or "Kyoto"
    area_hint = "Taiwan" if region == "tw" else "Japan"
    qk = _extract_search_category_keywords(query)
    slots = _requested_meal_slots(query)
    out: list[str] = [
        f"restaurants dining food {locality}",
        f"popular local food near {locality} station",
        f"walkable restaurants downtown {locality}",
    ]
    if qk:
        out.append(f"{' '.join(qk[:8])} {locality} {area_hint}")
    for sl in slots:
        bundle = _SLOT_FORCED_PLACES_QUERY.get(sl)
        if bundle:
            out.append(f"{bundle} {locality} surrounding area")
    seen: set[str] = set()
    uniq: list[str] = []
    for line in out:
        key = " ".join(line.split()).lower()
        if key and key not in seen:
            seen.add(key)
            uniq.append(line.strip())
    return uniq[:12]


async def _llm_broad_geo_search_queries(
    user_query: str,
    city: str,
    region: str,
    *,
    chat_client: OpenAIChatCompletionClient | None = None,
) -> tuple[list[str], str]:
    """
    Ask an LLM for broader English Places search phrases; fall back to heuristics.
    Returns (queries, mode_tag for audit).
    """
    fb = _fallback_broad_geo_queries(user_query, city, region)
    client = chat_client if chat_client is not None else _default_openai_chat_client
    if not client.is_configured():
        return fb, "heuristic_fallback_no_api_key"

    system_msg = (
        "You fix sparse Google Places Text Search results. Reply with JSON only, no markdown: "
        '{"queries":["..."]} containing 5 to 8 short English search strings. '
        "Broaden geography (neighborhoods, wards, stations, surrounding areas near the city); "
        "relax overly narrow cuisine filters. Do not repeat the same phrase twice."
    )
    user_msg = f"City: {city} (region={region}). User query:\n{(user_query or '')[:2000]}"
    try:
        payload = await client.chat_completion_payload(
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.35,
            max_tokens=450,
        )
        txt = str(payload["choices"][0]["message"]["content"] or "").strip()
        if txt.startswith("```"):
            txt = re.sub(r"^```(?:json)?\s*", "", txt)
            txt = re.sub(r"\s*```$", "", txt)
        obj = json.loads(txt)
        qs = obj.get("queries") if isinstance(obj, dict) else None
        if isinstance(qs, list):
            cleaned = [str(x).strip() for x in qs if str(x).strip()]
            if cleaned:
                return cleaned[:12], "openai_chat_json"
    except Exception:
        pass
    return fb, "heuristic_fallback_after_llm_error"


def _clamp_hhmm_token(hhmm: str) -> str | None:
    try:
        h, mm = [int(x) for x in hhmm.strip().split(":")]
        h = max(0, min(23, h))
        mm = max(0, min(59, mm))
        return f"{h:02d}:{mm:02d}"
    except Exception:
        return None


def _infer_close_time(opening_hours_today: str) -> str:
    text = (opening_hours_today or "").strip()
    if not text:
        return "22:00"
    for pat in (r"(\d{1,2}:\d{2})\s*[-–~]\s*(\d{1,2}:\d{2})", r"(\d{1,2}:\d{2}).*(\d{1,2}:\d{2})"):
        m = re.search(pat, text)
        if not m:
            continue
        hhmm = m.group(2)
        got = _clamp_hhmm_token(hhmm)
        if got:
            return got
    return "22:00"


def _infer_open_time(opening_hours_today: str) -> str:
    """Parse opening start from the same human-readable string as Google weekday_text / API summaries."""
    text = (opening_hours_today or "").strip()
    if not text:
        return "11:00"
    for pat in (r"(\d{1,2}:\d{2})\s*[-–~]\s*(\d{1,2}:\d{2})", r"(\d{1,2}:\d{2}).*(\d{1,2}:\d{2})"):
        m = re.search(pat, text)
        if not m:
            continue
        hhmm = m.group(1)
        got = _clamp_hhmm_token(hhmm)
        if got:
            return got
    return "11:00"


def _resolve_dynamic_open_time(place: dict, opening_hours_today: str) -> str:
    raw = place.get("open_time")
    if raw is not None:
        s = str(raw).strip()
        if s:
            got = _clamp_hhmm_token(s)
            if got:
                return got
    return _infer_open_time(opening_hours_today)


def _is_dynamic_time_unknown(place: dict, opening_hours_today: str) -> bool:
    raw = place.get("open_time")
    if raw is not None and _clamp_hhmm_token(str(raw).strip()):
        return False
    return _infer_open_time(opening_hours_today) == "11:00" and not (opening_hours_today or "").strip()


def _build_dynamic_shop_profile(place: dict, region: str) -> ShopProfile:
    name = str(place.get("name", "")).strip() or "Unknown Place"
    addr = str(place.get("formatted_address", "")).strip()
    types = [str(x).lower() for x in place.get("types", [])]
    rating = float(place.get("rating", 0.0) or 0.0)
    opening_hours_today = str(place.get("opening_hours_today", "") or "")
    close_time = _infer_close_time(opening_hours_today)
    open_time = _resolve_dynamic_open_time(place, opening_hours_today)
    time_unknown = _is_dynamic_time_unknown(place, opening_hours_today)
    tags: list[str] = ["dynamic", "restaurant"]
    if time_unknown:
        tags.append("TIME_UNKNOWN")
    if "cafe" in name.lower() or any("cafe" in t for t in types):
        tags.extend(["cafe", "light"])
    if "ramen" in name.lower():
        tags.extend(["ramen", "main_meal"])
    occasion_tags = {"main_meal"}
    if "cafe" in tags:
        occasion_tags.update({"tea", "afternoon_tea"})
    if "ramen" in tags:
        occasion_tags.update({"lunch", "dinner"})
    trust = max(0.55, min(0.95, rating / 5.0 if rating > 0 else 0.68))
    review_texts = [str(x).strip() for x in place.get("reviews", []) if str(x).strip()]
    authority_blob = " ".join(
        [
            name,
            addr,
            " ".join(review_texts),
        ]
    ).lower()
    medal = ""
    if ("百名店" in authority_blob) or ("tabelog" in authority_blob and "100" in authority_blob):
        medal = "百名店"
    michelin_star = 0
    if ("米其林" in authority_blob) or ("michelin" in authority_blob):
        michelin_star = 1
    lineage: list[str] = []
    if any(k in authority_blob for k in ("系譜", "lineage", "伝承", "傳承", "監修", "修業")):
        lineage = ["dynamic_lineage_detected"]
    if not review_texts:
        seeded: list[str] = []
        if michelin_star > 0:
            seeded.append("評論提及米其林。")
        if medal:
            seeded.append("評論提及百名店。")
        if lineage:
            seeded.append("評論提及料理系譜與傳承。")
        review_texts = seeded
    return ShopProfile(
        name=name,
        close_time=close_time,
        open_time=open_time,
        booking_type=BookingType.NONE,
        queue_strategy=QueueStrategy.PHYSICAL_LINE,
        last_call_offset=30,
        is_cash_only=False,
        sns_handle=re.sub(r"[^a-zA-Z0-9_]", "_", name.lower())[:40],
        avg_eat_minutes=50,
        is_famous=rating >= 4.4,
        base_wait_minutes=20 if rating >= 4.2 else 28,
        base_health_impact=0.55,
        customization_score=0.55,
        flavor_intensity=0.5,
        flavor_category=FlavorCategory.LIGHT if "light" in tags or "cafe" in tags else FlavorCategory.HEAVY,
        flavor_vector={"salt": 0.4, "fat": 0.35, "umami": 0.55, "acid": 0.25, "spice": 0.2},
        min_lead_hours=0,
        trust_score=trust,
        source_scores={"google": trust},
        authority_data=AuthorityData(
            tablelog_medal=medal,
            michelin_star=michelin_star,
            chef_lineage=lineage,
            specialty_items=[],
            google_reviews=review_texts[:6],
            review_count=max(len(review_texts), int(round(rating * 40))) if rating > 0 else len(review_texts),
        ),
        tags=tags,
        occasion_tags=occasion_tags,
        neighborhood=addr.split(",")[0] if addr else name,
        region=region,
        latitude=float(place.get("lat", 0.0) or 0.0),
        longitude=float(place.get("lng", 0.0) or 0.0),
        google_rating=rating,
        open_now=place.get("open_now", None),
        opening_hours_today=opening_hours_today,
    )


def _dynamic_pool_row_from_place(place: dict, region: str) -> dict:
    """Normalize a Places result dict into the same structure as node_food_search dynamic_shop_pool rows."""
    prof = _build_dynamic_shop_profile(place, region=region)
    time_unknown = "TIME_UNKNOWN" in prof.tags
    return {
        "name": prof.name,
        "lat": prof.latitude,
        "lng": prof.longitude,
        "rating": prof.google_rating,
        "open_now": prof.open_now,
        "opening_hours_today": prof.opening_hours_today,
        "formatted_address": place.get("formatted_address", ""),
        "types": place.get("types", []),
        "reviews": place.get("reviews", []),
        "open_time": prof.open_time,
        "time_unknown": time_unknown,
        "region": region,
    }




def _reliability_cutoff_for_region(region: str) -> float:
    # Taiwan has denser Google coverage; avoid over-penalizing normal variance.
    if region == "tw":
        return 53.0
    return 60.0


@traced
def node_clarify_constraint(state: AgentState) -> AgentState:
    """Prompt strict vs loose for single-hint dietary exclusions before retrieval (non-revision only)."""
    if state.get("error"):
        return state
    intent = state.get("intent") or {}
    if not isinstance(intent, dict):
        intent = {}
    if not intent:
        state.pop("awaiting_dietary_clarification", None)
        state.pop("clarification_broadcast", None)
        return state
    resolved = dict(state.get("dietary_clarification_resolved") or {})
    hint = _ambiguous_dietary_hint_for_clarification(intent, resolved)
    if hint is None:
        state.pop("awaiting_dietary_clarification", None)
        state.pop("clarification_broadcast", None)
        return state
    if intent.get("is_revision"):
        return state
    qbody = _DIETARY_CLARIFICATION_QUESTIONS.get(hint)
    if not qbody:
        return state
    state["pending_dietary_clarification"] = {
        "hint": hint,
        "intent_snapshot": copy.deepcopy(intent),
    }
    state["awaiting_dietary_clarification"] = True
    state["clarification_broadcast"] = {"type": "clarification", "question": qbody, "hint": hint}
    state.setdefault("transit_audit", []).append(_dj("dietary_clarification_prompt", hint=hint))
    return state


@traced
def node_route_intent(state: AgentState) -> AgentState:
    """Parse intent once, store in state, and set routing flags."""
    q = state.get("query", "") or ""
    if _consume_pending_dietary_clarification_answer(state):
        return state
    ctid = (state.get("checkpoint_thread_id") or "").strip()
    prev_snap = state.get("intent")
    hist_full = list(state.get("intent_history") or [])
    prev_model = None
    if isinstance(prev_snap, dict) and prev_snap:
        try:
            prev_model = _intent_from_snapshot_dict(prev_snap)
        except Exception:
            prev_model = None
    elif ctid and hist_full:
        last_snap = hist_full[-1]
        if isinstance(last_snap, dict):
            try:
                prev_model = _intent_from_snapshot_dict(last_snap)
            except Exception:
                prev_model = None
    hist = list(hist_full)
    if isinstance(prev_snap, dict) and prev_snap:
        hist.append(copy.deepcopy(prev_snap))
    try:
        intent = _parse_intent(
            q,
            _svc_llm_router(state),
            user_locale=state.get("user_locale") or None,
            user_lat=state.get("user_lat"),
            user_lng=state.get("user_lng"),
            previous_intent=prev_model,
            prev_itinerary=str(state.get("prev_itinerary") or ""),
        )
    except Exception as exc:
        _attach_node_error(state, "route_intent", exc)
        state["intent_history"] = hist
        state["intent"] = None
        state["plan_excluded_shop_tags"] = []
        state.setdefault("research_log", []).append(
            _dj("intent_parse_failed", reason=str(exc)[:500])
        )
        state["wants_flight_search"] = False
        return state
    state["intent_history"] = hist
    state["intent"] = intent.as_dict()
    if intent.is_revision:
        qtrim = q.strip()
        preview = qtrim if len(qtrim) <= 160 else qtrim[:160] + "⋯"
        msg = (
            f"對話線程內調整意向；請求摘要：{preview}\n→ city={intent.city} region={intent.region} tags={intent.category_tags}"
        )
        state.setdefault("research_log", []).append(
            "使用者修正意圖：" + audit_json_line_as_text(_dj("user_intent_revision", message=msg))
        )
    state["plan_excluded_shop_tags"] = sorted(
        _build_plan_excluded_shop_tags(q, state["intent"], state.get("dietary_profile"))
    )
    pend = state.get("pending_dietary_clarification")
    if isinstance(pend, dict) and pend.get("hint"):
        cur_amb = _ambiguous_dietary_hint_for_clarification(
            state["intent"] or {}, state.get("dietary_clarification_resolved") or {}
        )
        if cur_amb != pend.get("hint"):
            state.pop("pending_dietary_clarification", None)
            state.pop("awaiting_dietary_clarification", None)
            state.pop("clarification_broadcast", None)
    # wants_flight_search retained for context but flight booking is not in main flow
    state["wants_flight_search"] = intent.wants_flight
    return state


@traced
def node_flight_search(state: AgentState) -> AgentState:
    """DEPRECATED: Duffel flight-search node — removed from main graph in Task 7.

    Full implementation lives in git history.  Kept here as a 1-line reference for
    duffel.py / saga.py / acl.py distributed-transaction pattern.
    To re-enable: add back to build_graph() and wire route_intent → flight_search → retriever.
    """
    # Reference: _svc_flight_service(state).search_offers(origin, dest, date) …
    # normalise the response dict, log to research_log, then pass to node_audit (Saga).
    state["research_log"].append(
        _dj("flight_search_skipped", reason="node_flight_search not in main graph (Task 7)")
    )
    return state


@traced
def node_food_search(state: AgentState) -> AgentState:
    """DEPRECATED: Pure-heuristic Places search — replaced by node_retriever (Task 4, Task 7).

    node_retriever uses RetrieverAgent + LLMRouter (Gemini) for candidate reasoning.
    This heuristic fallback is kept as a reference for the NearbySearchTool integration.
    Pattern: _plan_dynamic_place_queries → NearbySearchTool.search_places → dynamic_shop_pool.
    """
    state["research_log"].append(
        _dj("food_search_skipped", reason="node_food_search not in main graph; use node_retriever")
    )
    return state


@trace_agent_stage("retriever")
async def node_retriever(state: AgentState) -> AgentState:
    """LLM-powered retrieval node: discovers candidates + produces reasoning notes.

    Replaces the pure-heuristic node_food_search in the main graph path.
    Also back-fills state["dynamic_shop_pool"] so node_plan remains compatible.
    """
    if state.get("error"):
        return state
    print(_dj("debug_print", node="node_retriever", message="RetrieverAgent starting"))
    try:
        agent = RetrieverAgent(llm_router=_svc_llm_router(state))
        report = await agent.arun(state)
    except Exception as exc:
        _attach_node_error(state, "retriever", exc)
        state.setdefault("research_log", []).append(_dj("retriever_failed", reason=str(exc)[:800]))
        return state

    # Append to append-only retrieval_history
    state["retrieval_history"] = list(state.get("retrieval_history") or []) + [report.as_dict()]

    # Back-fill dynamic_shop_pool for backward compat with node_plan
    _intent = state.get("intent") or {}
    region = _intent.get("region") or "jp"
    state["dynamic_shop_pool"] = [
        {
            "name": s.name,
            "lat": s.latitude,
            "lng": s.longitude,
            "rating": s.google_rating,
            "open_time": getattr(s, "open_time", "11:00"),
            "opening_hours_today": getattr(s, "opening_hours_today", ""),
            "open_now": False,
            "time_unknown": "TIME_UNKNOWN" in (s.tags or []),
            "region": region,
        }
        for s in report.candidates
        if "dynamic" in (s.tags or [])
    ]

    state["research_log"].append(
        _dj(
            "retriever_complete",
            city=report.city,
            seed_count=report.seed_count,
            dynamic_count=report.dynamic_count,
            total_candidates=len(report.candidates),
            notes_count=len(report.notes),
            gaps=report.gaps,
        )
    )
    return state


def _researcher_shop_pool(state: AgentState) -> list[ShopProfile]:
    excluded = plan_excluded_frozenset(state)
    _intent = state.get("intent") or {}
    _city = _intent.get("city") or ""
    if _city == "台北":
        seed = list(_build_shop_catalog_taipei())
    elif _city == "東京":
        seed = list(_build_shop_catalog_tokyo())
    else:
        seed = list(_build_shop_catalog())
    dyn: list[ShopProfile] = []
    for p in list(state.get("dynamic_shop_pool", []) or [])[:60]:
        dyn.append(_build_dynamic_shop_profile(p, region=str(p.get("region", "jp"))))
    out: list[ShopProfile] = []
    seen: set[str] = set()
    for s in seed + dyn:
        if s.name in seen:
            continue
        seen.add(s.name)
        out.append(s)
    return filter_shop_profiles_by_dietary_exclusions(out, excluded)


def _researcher_seed_catalog(state: AgentState) -> list[ShopProfile]:
    """City-scoped seed list only (no dynamic pool)."""
    excluded = plan_excluded_frozenset(state)
    _intent = state.get("intent") or {}
    _city = _intent.get("city") or ""
    if _city == "台北":
        seed = list(_build_shop_catalog_taipei())
    elif _city == "東京":
        seed = list(_build_shop_catalog_tokyo())
    else:
        seed = list(_build_shop_catalog())
    return filter_shop_profiles_by_dietary_exclusions(seed, excluded)


async def _call_researcher_prompt(
    query: str,
    shops: list[ShopProfile],
    *,
    meal_slots: list[str],
    seed_shops: list[ShopProfile],
    auditor_feedback: str,
    iteration: int,
    chat_client: OpenAIChatCompletionClient | None = None,
) -> tuple[list[str], str]:
    """Prompt-driven candidate generation. Falls back to deterministic heuristic."""

    cap_llm = max(8, len(meal_slots) + 2, 5)

    def _finalize(base: list[str], notes_tag: str) -> tuple[list[str], str]:
        merged, extras = _researcher_finalize_candidate_names(
            meal_slots=meal_slots,
            shops=shops,
            seed_shops=seed_shops,
            query=query,
            base_names=base,
            auditor_feedback=auditor_feedback,
            notes_suffix=notes_tag,
        )
        return merged, extras

    client = chat_client if chat_client is not None else _default_openai_chat_client
    sample = [
        {
            "name": s.name,
            "tags": list(s.tags),
            "occasion_tags": sorted(_seed_shop_tag_bag(s)),
            "open_time": s.open_time,
            "close_time": s.close_time,
            "lat": s.latitude,
            "lng": s.longitude,
            "rating": s.google_rating,
        }
        for s in shops[: min(36, len(shops))]
    ]
    if client.is_configured():
        try:
            sys_msg = (
                "You are Researcher Agent (student). Build a foodie itinerary candidate list. "
                "You MUST nominate at least one distinct shop name per requested meal_slots entry "
                "(breakfast,lunch,tea,dinner,late_night,custom) drawn from shops when plausible. "
                'Return JSON only: {"candidate_names":[...],"notes":"..."}. '
                "Prefer higher ratings/trust while covering every slot requested. "
                "Respect auditor feedback if provided."
            )
            user_msg = json.dumps(
                {
                    "query": query,
                    "meal_slots": meal_slots,
                    "iteration": iteration,
                    "auditor_feedback": auditor_feedback,
                    "shops": sample,
                },
                ensure_ascii=False,
            )
            resp_json = await client.chat_completion_payload(
                messages=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.2,
                max_tokens=520,
            )
            txt = str(resp_json["choices"][0]["message"]["content"] or "").strip()
            txt = re.sub(r"^```(?:json)?\s*", "", txt)
            txt = re.sub(r"\s*```$", "", txt)
            obj = json.loads(txt)
            names = [str(x).strip() for x in obj.get("candidate_names", []) if str(x).strip()]
            notes = str(obj.get("notes", "") or "")
            shop_ok = {s.name for s in shops}
            filtered = [n for n in names if n in shop_ok]
            if filtered:
                return _finalize(filtered[:cap_llm], f"openai_researcher;{notes}")
        except Exception:
            pass

    # Deterministic heuristic: anchors from seed catalog + greedy top-up from combined pool.
    return _finalize([], "heuristic_fallback_researcher")


@traced
async def node_researcher(state: AgentState) -> AgentState:
    if state.get("error"):
        return state
    try:
        shops = _researcher_shop_pool(state)
        seed_shops = _researcher_seed_catalog(state)
        query = state.get("query", "") or ""
        meal_slots = _effective_plan_meal_slots(state.get("intent") or {}, query)
        iteration = int(state.get("research_iteration", 0)) + 1
        names, notes = await _call_researcher_prompt(
            query,
            shops,
            meal_slots=meal_slots,
            seed_shops=seed_shops,
            auditor_feedback=state.get("auditor_feedback", "") or "",
            iteration=iteration,
            chat_client=_svc_openai_chat_client(state),
        )
        state["research_iteration"] = iteration
        state["researcher_candidate_names"] = names
        state["researcher_notes"] = notes
        state["transit_audit"].append(
            _dj("researcher_iteration", iteration=iteration, candidate_count=len(names), notes=notes)
        )
        return state
    except Exception as exc:
        _attach_node_error(state, "researcher", exc)
        state.setdefault("transit_audit", []).append(_dj("researcher_failed", reason=str(exc)[:800]))
        return state


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    import math

    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    d1 = math.radians(lat2 - lat1)
    d2 = math.radians(lon2 - lon1)
    a = math.sin(d1 / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(d2 / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


@trace_agent_stage("critic")
def node_critic(state: AgentState) -> AgentState:
    """Run :class:`CriticAgent` over the retrieval pool; fills ``critique_history`` for routing."""
    intent = state.get("intent")

    def get_intent_field(obj: Any, key: str) -> Any:
        if obj is None:
            return None
        if isinstance(obj, dict):
            return obj.get(key)
        return getattr(obj, key, None)

    city = get_intent_field(intent, "city")

    # Only treat missing city as a hard block (serialized intent is a dict; meal_slots → CriticAgent).
    if not intent or not city or city == "unknown":
        msg = "為了精準規劃，我還需要知道您想去哪個城市？"
        state["auditor_rejected"] = True
        state["auditor_feedback"] = msg
        state["critique_history"] = list(state.get("critique_history") or []) + [
            {
                "verdict": "request_more",
                "accepted_names": [],
                "rejected_with_reason": [],
                "requests_for_retriever": [],
                "score_table": {},
                "fame_damped_table": {},
                "accolade_table": {},
                "llm_analysis": msg,
                "iteration": int(state.get("research_iteration", 0)),
            }
        ]
        return state

    if state.get("error"):
        return state
    print(_dj("debug_print", node="node_critic", message="CriticAgent starting"))
    try:
        agent = CriticAgent(llm_router=_svc_llm_router(state))
        report = agent.run(state)
    except Exception as exc:
        _attach_node_error(state, "critic", exc)
        state.setdefault("transit_audit", []).append(_dj("critic_failed", reason=str(exc)[:800]))
        return state

    state["critique_history"] = list(state.get("critique_history") or []) + [report.as_dict()]

    # Backward-compat: set auditor_rejected so the existing conditional edge works.
    # "deadlock" is NOT a rejection — skip the retry loop, fall through to plan.
    state["auditor_rejected"] = report.verdict == "request_more"
    state["auditor_feedback"] = "; ".join(report.requests_for_retriever)

    state.setdefault("transit_audit", []).append(
        _dj(
            "critic_verdict",
            verdict=report.verdict,
            accepted=len(report.accepted),
            rejected=len(report.rejected_with_reason),
            requests=report.requests_for_retriever,
            iteration=report.iteration,
            llm_analysis=report.llm_analysis,
        )
    )
    return state


@traced
def node_auditor(state: AgentState) -> AgentState:
    """DEPRECATED: Rule-based professor auditor — replaced by node_critic (CriticAgent, Task 5).

    Heuristic rules: operating-time boundary, inter-shop distance > 8 km, avg-rating < 4.1.
    Replaced by an LLM-powered CriticAgent (Claude via LLMRouter) that covers the same
    physical boundaries PLUS foodie-taste reasoning (marketing noise, ScoringEngine scores).
    """
    state["transit_audit"].append(
        _dj("auditor_review_skipped", reason="node_auditor not in main graph; use node_critic")
    )
    return state


@traced
def node_audit(state: AgentState) -> AgentState:
    """DEPRECATED: Saga 2-phase flight reservation — removed from main graph in Task 7.

    Full implementation: SagaEngine(kind="transactional").run([reserve_leg1, reserve_leg2]).
    Pattern: backward recovery — cancel_leg1 if reserve_leg2 fails.
    See saga.py, duffel.py, acl.py::ActionOutcome for the complete reference.
    See also: tests/legacy/test_saga_compensation.py, test_idempotency.py, test_resilience.py.
    """
    state["transit_audit"].append(
        _dj("node_audit_skipped", reason="node_audit not in main graph (Task 7 ADR)")
    )
    return state


def _combine_itinerary_clock(itinerary_start: datetime, t: datetime) -> datetime:
    """Place `t`'s wall-clock onto `itinerary_start`'s calendar day (same tz semantics as start)."""
    if itinerary_start.tzinfo is None and t.tzinfo is not None:
        t = t.replace(tzinfo=None)
    elif itinerary_start.tzinfo is not None and t.tzinfo is None:
        t = t.replace(tzinfo=itinerary_start.tzinfo)
    return itinerary_start.replace(
        hour=t.hour,
        minute=t.minute,
        second=t.second,
        microsecond=t.microsecond,
    )


def _relay_layer_one_edges_inplace(
    graph: SpatioTemporalGraph,
    *,
    ranked: list[RankedShop],
    traffic,
) -> None:
    """Mirror GraphBuilder layer-1 edge rule after mutating node datetimes (decision_engine stays unchanged)."""
    ranked_index = {r.shop.name: r for r in ranked}
    shop_index = {r.shop.name: r.shop for r in ranked}
    by_slot: dict[int, list] = {}
    for n in graph.nodes or []:
        by_slot.setdefault(n.slot_index, []).append(n)
    if not by_slot:
        graph.edges = []
        graph.debug_traces = []
        return
    slot_count = max(by_slot.keys()) + 1
    edges: list[GraphEdge] = []
    debug_traces: list[str] = []
    if slot_count > 1:
        for i in range(slot_count - 1):
            from_nodes = by_slot.get(i, [])
            if not from_nodes:
                continue
            for j in range(i + 1, slot_count):
                to_nodes = by_slot.get(j, [])
                if not to_nodes:
                    continue
                for a in from_nodes:
                    shop_a = shop_index.get(a.shop_name)
                    if shop_a is None:
                        continue
                    for b in to_nodes:
                        if a.shop_name == b.shop_name:
                            continue
                        shop_b = shop_index.get(b.shop_name)
                        if shop_b is None:
                            continue
                        travel_m = 18 + (8 * i)
                        fastest_finish_at = a.start_time + timedelta(
                            minutes=int(shop_a.base_wait_minutes)
                            + int(shop_a.min_eat_minutes or shop_a.avg_eat_minutes)
                        )
                        ready_at = fastest_finish_at + timedelta(minutes=travel_m)
                        open_b = ItinerarySynthesizer._shop_open_at(b.start_time, shop_b)
                        if ready_at <= b.start_time and b.start_time >= open_b:
                            status = traffic.get_route_status(a.shop_name, b.shop_name)
                            queue_risk = min(1.0, max(0.0, float(shop_b.base_wait_minutes) / 45.0))
                            slack_m = max(0.0, (b.start_time - ready_at).total_seconds() / 60.0)
                            expected_buffer_m = max(0.0, float(status.transport_buffer_minutes))
                            buffer_gap_m = max(0.0, expected_buffer_m - slack_m)
                            travel_buffer_gap = min(1.0, buffer_gap_m / 45.0)
                            base_score = float(ranked_index[b.shop_name].final_score)
                            weight = ScoringEngine.risk_adjusted_score(
                                base_score,
                                queue_risk=queue_risk,
                                travel_buffer_gap=travel_buffer_gap,
                            )
                            edges.append(GraphEdge(from_node_id=a.node_id, to_node_id=b.node_id, weight=weight))
                        else:
                            reasons: list[str] = []
                            if ready_at > b.start_time:
                                reasons.append(
                                    f"準備時間 (ready_at) {ready_at.strftime('%H:%M')} > 開始時間 (B.start) {b.start_time.strftime('%H:%M')}"
                                )
                            if b.start_time < open_b:
                                reasons.append(
                                    f"B.start {b.start_time.strftime('%H:%M')} < B.open_time {open_b.strftime('%H:%M')}"
                                )
                            reason_text = "；".join(reasons) if reasons else "未知原因"
                            debug_traces.append(
                                _dj(
                                    "graph_rejected_edge",
                                    from_shop=a.shop_name,
                                    to_shop=b.shop_name,
                                    detail=reason_text,
                                )
                            )
    graph.edges = edges
    graph.debug_traces = debug_traces


def _pin_dp_graph_to_itinerary_day_and_relayer_edges(
    graph: SpatioTemporalGraph,
    *,
    itinerary_start: datetime,
    ranked: list[RankedShop],
    traffic,
) -> None:
    """DP graph nodes must share one excursion calendar day — align per-shop align() roll-forward with itinerary_start."""
    if not graph.nodes:
        return
    for n in graph.nodes:
        dur = n.end_time - n.start_time
        n.start_time = _combine_itinerary_clock(itinerary_start, n.start_time)
        n.end_time = n.start_time + dur
    _relay_layer_one_edges_inplace(graph, ranked=ranked, traffic=traffic)


def _expand_graph_node_tags_from_shop_profiles(graph: SpatioTemporalGraph, ranked: list[RankedShop]) -> None:
    """DP only sees GraphNode.tags; merge occasion_tags for breakfast/affinity without editing decision_engine GraphBuilder."""
    by_name = {r.shop.name: r.shop for r in ranked}
    for n in graph.nodes or []:
        sp = by_name.get(n.shop_name)
        if sp is None:
            continue
        bag = {str(x).lower() for x in sp.tags} | {str(x).lower() for x in getattr(sp, "occasion_tags", ()) or []}
        n.tags = tuple(sorted(bag))


def agent_dp_find_optimal_path_no_shop_repeat(
    graph: SpatioTemporalGraph,
    required_length: int,
    must_have_tags: set[str] | None = None,
    banned_node_ids: set[str] | None = None,
    solver_audit_log: list[str] | None = None,
    excluded_shop_tags: frozenset[str] | None = None,
) -> list[GraphNode]:
    """Same semantics as decision_engine DP, plus distinct shop constraint + soft slot-tag affinity."""
    if solver_audit_log is not None:
        solver_audit_log.append(
            _dj(
                "dp_start_agent",
                variant="unique_shops_soft_slot_tag_affinity",
                required_length=required_length,
                nodes=len(graph.nodes or []),
            )
        )
    if required_length <= 0 or not graph.nodes:
        if solver_audit_log is not None:
            solver_audit_log.append(_dj("dp_early_exit_agent", reason="invalid_required_length_or_empty_graph"))
        return []
    must_have_tags_l = {t.lower() for t in (must_have_tags or set())}
    slot_names_raw = getattr(graph, "_agent_meal_slot_names_for_dp", None)
    slot_names: list[str] | None = list(slot_names_raw) if isinstance(slot_names_raw, list) else None

    required_tags_list = sorted(must_have_tags_l)
    tag_idx = {t: i for i, t in enumerate(required_tags_list)}
    full_mask = (1 << len(required_tags_list)) - 1

    banned_node_ids_set = banned_node_ids or set()
    usable_nodes = [n for n in graph.nodes if n.node_id not in banned_node_ids_set]
    _excl = excluded_shop_tags or frozenset()
    if _excl:
        usable_nodes = [
            n
            for n in usable_nodes
            if not ItinerarySynthesizer.node_tags_intersect_excluded(n.tags, _excl)
        ]
    if not usable_nodes:
        if solver_audit_log is not None:
            solver_audit_log.append(_dj("dp_early_exit_agent", reason="all_nodes_filtered_by_banned_node_ids"))
        return []
    node_by_id = {n.node_id: n for n in usable_nodes}
    indeg: dict[str, int] = {nid: 0 for nid in node_by_id}
    out_edges: dict[str, list[GraphEdge]] = {nid: [] for nid in node_by_id}
    for e in graph.edges:
        if (
            e.from_node_id not in node_by_id
            or e.to_node_id not in node_by_id
            or e.from_node_id in banned_node_ids_set
            or e.to_node_id in banned_node_ids_set
        ):
            continue
        indeg[e.to_node_id] += 1
        out_edges[e.from_node_id].append(e)

    queue = [nid for nid, d in indeg.items() if d == 0]
    topo: list[str] = []
    while queue:
        queue.sort(key=lambda nid: (node_by_id[nid].slot_index, node_by_id[nid].start_time))
        cur = queue.pop(0)
        topo.append(cur)
        for e in out_edges[cur]:
            indeg[e.to_node_id] -= 1
            if indeg[e.to_node_id] == 0:
                queue.append(e.to_node_id)
    if len(topo) < len(node_by_id):
        topo = sorted(node_by_id.keys(), key=lambda nid: (node_by_id[nid].slot_index, node_by_id[nid].start_time))

    def tag_mask(node: GraphNode) -> int:
        m = 0
        node_tags_low = {t.lower() for t in node.tags}
        for t, idx in tag_idx.items():
            if t in node_tags_low:
                m |= 1 << idx
        return m

    def node_objective_score(node: GraphNode) -> float:
        ideal = max(1, int(node.ideal_duration_minutes))
        actual = max(1, int(node.actual_duration_minutes))
        fidelity = max(0.0, min(1.0, float(actual) / float(ideal)))
        bonus = 1.0
        if slot_names and 0 <= node.slot_index < len(slot_names):
            slot_nm = str(slot_names[node.slot_index]).lower()
            preferred = ItinerarySynthesizer.SLOT_PREFERRED_TAGS.get(slot_nm)
            if preferred and {str(t).lower() for t in node.tags} & preferred:
                bonus = 1.3
        return float(node.final_score) * fidelity * bonus

    # Key: (node_id, path_len, must_have_bitmask, frozenset(shop_names_on_path)).
    KeyT = tuple[str, int, int, frozenset[str]]
    best: dict[KeyT, float] = {}
    prev: dict[KeyT, KeyT | None] = {}

    for nid in topo:
        node = node_by_id[nid]
        m = tag_mask(node)
        fshops = frozenset({node.shop_name})
        key = (nid, 1, m, fshops)
        best[key] = node_objective_score(node)
        prev[key] = None

    for nid in topo:
        outgoing = out_edges.get(nid, [])
        cur_states = [(k, v) for k, v in best.items() if k[0] == nid]
        if not cur_states:
            continue
        for cur_key, cur_score in cur_states:
            _, cur_len, cur_mask, fshops_cur = cur_key
            if cur_len >= required_length:
                continue
            for e in outgoing:
                to_node = node_by_id[e.to_node_id]
                if to_node.shop_name in fshops_cur:
                    continue
                next_mask = cur_mask | tag_mask(to_node)
                fshops_next = frozenset(fshops_cur | {to_node.shop_name})
                nxt: KeyT = (e.to_node_id, cur_len + 1, next_mask, fshops_next)
                cand = cur_score + node_objective_score(to_node)
                if cand > best.get(nxt, float("-inf")):
                    best[nxt] = cand
                    prev[nxt] = cur_key

    max_len = max((k[1] for k in best.keys()), default=0)
    target_len = required_length if any(k[1] == required_length for k in best.keys()) else max_len
    if target_len <= 0:
        if solver_audit_log is not None:
            solver_audit_log.append(_dj("dp_early_exit_agent", reason="no_feasible_terminal_state"))
        return []
    terminal_keys = [k for k in best.keys() if k[1] == target_len]
    if required_tags_list:
        covered = [k for k in terminal_keys if k[2] == full_mask]
        if covered:
            terminal_keys = covered
        elif target_len == required_length:
            feasible_lens = sorted({k[1] for k in best.keys()}, reverse=True)
            for ln in feasible_lens:
                covered_ln = [k for k in best.keys() if k[1] == ln and k[2] == full_mask]
                if covered_ln:
                    terminal_keys = covered_ln
                    target_len = ln
                    break
    end_key = max(terminal_keys, key=lambda k: best[k])

    path_keys: list[KeyT] = []
    cur_k: KeyT | None = end_key
    while cur_k is not None:
        path_keys.append(cur_k)
        cur_k = prev.get(cur_k)
    path_keys.reverse()
    resolved_path = [node_by_id[k[0]] for k in path_keys]

    uniq = len({p.shop_name for p in resolved_path})
    if solver_audit_log is not None:
        solver_audit_log.append(
            _dj(
                "dp_path_selected",
                variant="agent_unique_shops",
                unique_shop_count=uniq,
                path=[{"shop": n.shop_name, "slot": n.slot_index} for n in resolved_path],
            )
        )
    return resolved_path


def _dp_graph_builder_with_itinerary_day_pin(
    ranked: list[RankedShop],
    traffic,
    start_time: datetime,
    *,
    meal_slots: list[str] | None = None,
    mode: OptimizationMode = OptimizationMode.BALANCED,
    requested_meal_count: int | None = None,
    slot_required_tags: dict[str, set[str]] | None = None,
    excluded_shop_tags: frozenset[str] | None = None,
) -> SpatioTemporalGraph:
    g = _RAW_GRAPHBUILDER_BUILD(
        ranked,
        traffic,
        start_time,
        meal_slots=meal_slots,
        mode=mode,
        requested_meal_count=requested_meal_count,
        slot_required_tags=slot_required_tags,
        excluded_shop_tags=excluded_shop_tags,
    )
    _pin_dp_graph_to_itinerary_day_and_relayer_edges(
        g, itinerary_start=start_time, ranked=ranked, traffic=traffic
    )
    setattr(
        g,
        "_agent_meal_slot_names_for_dp",
        list(ItinerarySynthesizer._normalize_slot_sequence(meal_slots or [])),
    )
    _expand_graph_node_tags_from_shop_profiles(g, ranked)
    return g


@contextmanager
def _pinned_travel_dp_calendar_and_solver():
    """Calendar-consistent graph timestamps + tag merge for DP (uses decision_engine.find_optimal_path during this block)."""
    GraphBuilder.build_graph = _dp_graph_builder_with_itinerary_day_pin
    try:
        yield
    finally:
        GraphBuilder.build_graph = _RAW_GRAPHBUILDER_BUILD


def _append_graph_physical_transition_audit(
    *,
    graph,
    ranked: list[RankedShop],
    traffic,
    transit_audit: list,
    mode: OptimizationMode,
) -> None:
    """Log physical edge math (matches GraphBuilder layer-1) + sample slot_i→slot_{i+1} probes."""
    shop_index = {r.shop.name: r.shop for r in ranked}
    traces = list(getattr(graph, "debug_traces", []) or [])
    rejects = 0
    reject_detail_sample: list[str] = []
    for tr in traces[:80]:
        try:
            o = json.loads(tr)
        except Exception:
            continue
        if isinstance(o, dict) and str(o.get("event")) == "graph_rejected_edge":
            rejects += 1
            if len(reject_detail_sample) < 4:
                dst = str(o.get("detail") or o.get("message") or "")[:260]
                reject_detail_sample.append(f"{o.get('from_shop')}→{o.get('to_shop')}: {dst}")

    by_slot: dict[int, list] = {}
    for n in graph.nodes or []:
        by_slot.setdefault(n.slot_index, []).append(n)
    probes: list[dict] = []
    indices = sorted(by_slot.keys())
    for low_i in range(max(0, len(indices) - 1)):
        i = indices[low_i]
        j = indices[low_i + 1]
        from_nodes = sorted(by_slot[i], key=lambda n: str(n.shop_name))[:5]
        to_nodes = sorted(by_slot[j], key=lambda n: str(n.shop_name))[:8]
        for a in from_nodes:
            shop_a = shop_index.get(a.shop_name)
            if shop_a is None:
                continue
            travel_m = 18 + (8 * i)
            fastest_finish = a.start_time + timedelta(
                minutes=int(shop_a.base_wait_minutes)
                + int(shop_a.min_eat_minutes or shop_a.avg_eat_minutes)
            )
            ready_physical = fastest_finish + timedelta(minutes=travel_m)
            digest_minutes = ItinerarySynthesizer._calculate_cooldown(
                shop_a,
                mode=mode,
                requested_meal_count=None,
                appetite_light_mode=False,
            )
            hypothetical_cool_ready = fastest_finish + timedelta(minutes=int(digest_minutes))
            for b in to_nodes[:3]:
                shop_b = shop_index.get(b.shop_name)
                if shop_b is None:
                    continue
                open_b = ItinerarySynthesizer._shop_open_at(b.start_time, shop_b)
                passed = ready_physical <= b.start_time and b.start_time >= open_b
                probes.append(
                    {
                        "slot_from": i,
                        "slot_to": j,
                        "from_shop": a.shop_name,
                        "from_start": str(a.start_time),
                        "from_end_ideal": str(getattr(a, "end_time", "")),
                        "eat_end_min_path": str(fastest_finish),
                        "cooldown_na_in_graph_edges": digest_minutes,
                        "hypothetical_ready_if_digest_enforced": str(hypothetical_cool_ready),
                        "travel_minutes_edge": travel_m,
                        "ready_at_physical_graph": str(ready_physical),
                        "to_shop": b.shop_name,
                        "b_start": str(b.start_time),
                        "b_open": str(open_b),
                        "edge_accepts_physical_layer1": passed,
                    }
                )
        if len(probes) >= 12:
            break

    slot_timelines: list[dict] = []
    for si in indices:
        layer = sorted(by_slot[si], key=lambda n: str(n.shop_name))
        if not layer:
            continue
        pick = layer[0]
        sa = shop_index.get(pick.shop_name)
        if sa is None:
            continue
        eat_end = pick.start_time + timedelta(
            minutes=int(sa.base_wait_minutes)
            + int(sa.min_eat_minutes or sa.avg_eat_minutes)
        )
        digest_min = ItinerarySynthesizer._calculate_cooldown(
            sa,
            mode=mode,
            requested_meal_count=None,
            appetite_light_mode=False,
        )
        travel_out = 18 + (8 * si)
        slot_timelines.append(
            {
                "slot_index": si,
                "repr_shop": pick.shop_name,
                "start": str(pick.start_time),
                "end_ideal_layer": str(pick.end_time),
                "eat_end_min_wait_path": str(eat_end),
                "synth_cooldown_minutes_note": digest_min,
                "outgoing_travel_minutes_layer1_formula": travel_out,
                "ready_at_physical_to_next_formula": (
                    f"eat_end_min + {travel_out}m travel (cooldown omitted in GraphBuilder)"
                ),
            }
        )

    row = _dj(
        "graph_physical_probe",
        unique_slots=len(by_slot),
        nodes=len(graph.nodes or []),
        edges=len(graph.edges or []),
        rejected_edges_logged=rejects,
        reject_reason_samples=reject_detail_sample,
        slot_repr_timeline=slot_timelines,
        sample_transitions=probes,
        note=(
            "ready_at=start+wait+min_eat+travel; GraphBuilder omits digestion cooldown vs synthesize()"
        ),
    )
    transit_audit.append(row)

    dbg = os.getenv("GRAPH_PHYSICS_DEBUG", "").strip().lower()
    if dbg not in {"", "0", "false"}:
        try:
            obj = json.loads(row)
            print(json.dumps({"GRAPH_PHYSICS_DEBUG": obj}, ensure_ascii=False, indent=2, default=str))
        except Exception:
            print(row)


@trace_agent_stage("planner")
async def node_plan(state: AgentState, config: Optional[RunnableConfig] = None) -> AgentState:
    print(_dj("debug_print", node="node_plan", message="Generating outcome report"))
    if state.get("error"):
        out = _finalize_plan_on_agent_error(state)
        extend_turn_checkpoint_in_state(out, config)
        return out
    try:
        out = await _node_plan_core(state)
    except Exception as exc:
        logger.exception("node_plan failed agent_run_id=%s", state.get("agent_run_id"))
        _attach_node_error(state, "plan", exc)
        out = _finalize_plan_on_agent_error(state)
    extend_turn_checkpoint_in_state(out, config)
    return out


async def _node_plan_core(state: AgentState) -> AgentState:
    query_text = state.get("query", "") or ""

    report = "## Travel Agent - Live Run\n\n"
    report += f"**Run ID:** `{state.get('agent_run_id','')}` "
    report += "(checkpoint persisted via LangGraph checkpointer)\n\n"
    if state.get("researcher_candidate_names"):
        report += "### Multi-Agent Loop\n"
        report += f"- Researcher iterations: {int(state.get('research_iteration', 0))}\n"
        report += f"- Candidate draft: {', '.join(state.get('researcher_candidate_names', []))}\n"
        report += f"- Critic feedback: {state.get('auditor_feedback', '')}\n\n"

    # Dynamic shop-aware planning block (ACL-style constraint check + fallback).
    _intent = state.get("intent") or {}
    report += "\n\n### Dynamic Shop Planning\n"
    if "appetite_light" in _intent.get("explicit_constraints", []):
        report += (
            "> **食量敏感**：已排除份量規定嚴格（strictness>0.9）且無「小盛／半份」選項的店家；"
            "含小盛選項者於排序中獲得加分。\n\n"
        )
    report += "| Slot | Shop | Outcome | Preparation Note |\n|---|---|---|---|\n"

    right_now_mode = _intent.get("mode") == "right_now"
    _tw = _intent.get("time_window") or [None, None]
    forced_start_hhmm = _tw[0] if _tw else None
    forced_end_hhmm = _tw[1] if _tw else None
    base_day = datetime.now(_APP_TZ)
    if forced_start_hhmm:
        sh, sm = [int(x) for x in forced_start_hhmm.split(":", 1)]
        # Hard rule: user-provided start overrides all default start-time sources.
        now = base_day.replace(hour=sh, minute=sm, second=0, microsecond=0)
        state["transit_audit"].append(
            _dj("global_window_start_enforced", start_hhmm=forced_start_hhmm)
        )
    else:
        now = base_day if right_now_mode else base_day.replace(hour=11, minute=30, second=0, microsecond=0)
    global_end_dt: datetime | None = None
    if forced_end_hhmm:
        eh, em = [int(x) for x in forced_end_hhmm.split(":", 1)]
        global_end_dt = now.replace(hour=eh, minute=em, second=0, microsecond=0)
        if global_end_dt <= now:
            global_end_dt = global_end_dt + timedelta(days=1)
        state["transit_audit"].append(_dj("global_window_end_enforced", end_hhmm=forced_end_hhmm))
    synth_start_time = now
    prefers_driving_mode = ("DRIVE" in state.get("query", "") or "自駕" in state.get("query", ""))
    dietary_raw = state.get("dietary_profile", {}) or {}
    inferred_ethics = _intent.get("dietary_hints")
    effective_ethics = str(dietary_raw.get("ethics", "omnivore"))
    if inferred_ethics and effective_ethics in {"unspecified", "omnivore", "regular", "none"}:
        effective_ethics = inferred_ethics
    advanced_mode = bool(state.get("advanced_mode", False))
    dietary_axis = DietaryAxis(
        ethics=effective_ethics,
        allergens={str(x) for x in dietary_raw.get("allergens", [])},
        religious=str(dietary_raw.get("religious", "none")),
        medical={str(x) for x in dietary_raw.get("medical", [])},
    )
    plan_excluded_tags = plan_excluded_frozenset(state)
    if not plan_excluded_tags and _intent:
        plan_excluded_tags = _build_plan_excluded_shop_tags(query_text, _intent, dietary_raw)
        state["plan_excluded_shop_tags"] = sorted(plan_excluded_tags)
    if plan_excluded_tags:
        state["transit_audit"].append(
            _dj(
                "plan_excluded_shop_tags",
                count=len(plan_excluded_tags),
                sample=sorted(plan_excluded_tags)[:24],
                note="ingress_via_route_intent_or_rebuilt",
            )
        )
    excluded_shop_names = normalized_excluded_shop_names_from_intent(_intent)
    if excluded_shop_names:
        state["transit_audit"].append(
            _dj(
                "plan_excluded_shop_names",
                count=len(excluded_shop_names),
                sample=list(excluded_shop_names)[:12],
            )
        )
    expand_city = _intent.get("city") or "京都"
    expand_region = _intent.get("region") or "jp"
    appetite_light_mode = "appetite_light" in _intent.get("explicit_constraints", [])
    explicit_category_tags = _plan_global_explicit_tags(query_text)
    must_have_tags = sorted(explicit_category_tags)
    meal_slots_eff = _effective_plan_meal_slots(_intent, query_text)
    sns_fixtures, traffic_fixtures = _region_driven_mock_fixtures(expand_city, expand_region)
    sns_provider = MockSnsProvider(fixtures=sns_fixtures)
    traffic_provider = MockTrafficProvider(fixtures=traffic_fixtures)
    state.setdefault("transit_audit", []).append(
        _dj(
            "region_fixture_loaded",
            city=expand_city,
            region=expand_region,
            sns_keys=sorted(sns_fixtures.keys()),
            traffic_edges=len(traffic_fixtures),
        )
    )
    dynamic_pool: list[dict] = list(state.get("dynamic_shop_pool", []) or [])
    if len(dynamic_pool) < _MIN_DYNAMIC_POOL_HEALTHY:
        state.setdefault("transit_audit", []).append(
            _dj(
                "dynamic_pool_low_warning",
                count=len(dynamic_pool),
                threshold=_MIN_DYNAMIC_POOL_HEALTHY,
                action="geo_expansion_llm_or_heuristic",
            )
        )
        expansion_queries, expansion_mode = await _llm_broad_geo_search_queries(
            query_text,
            expand_city,
            expand_region,
            chat_client=_svc_openai_chat_client(state),
        )
        must_for_expansion = sorted(_plan_global_explicit_tags(query_text))
        merged_pool: dict[str, dict] = {}
        for row in dynamic_pool:
            nk = (row.get("name") or "").strip().lower()
            if nk:
                merged_pool[nk] = dict(row)
        nearby_expand = NearbySearchTool()
        for qstr in expansion_queries:
            batch = nearby_expand.search_places(
                city=expand_city,
                user_query=qstr,
                limit=60,
                must_have_tags=must_for_expansion if must_for_expansion else None,
            )
            for p in batch:
                nk = (p.get("name") or "").strip().lower()
                if nk and nk not in merged_pool:
                    merged_pool[nk] = _dynamic_pool_row_from_place(p, expand_region)
            if len(merged_pool) >= 60:
                break
        dynamic_pool = list(merged_pool.values())[:60]
        state["dynamic_shop_pool"] = dynamic_pool
        state["transit_audit"].append(
            _dj(
                "dynamic_pool_expansion",
                mode=expansion_mode,
                query_batches=len(expansion_queries),
                merged_total=len(dynamic_pool),
            )
        )

    seed_shops: list[ShopProfile] = []
    dynamic_shops: list[ShopProfile] = []

    # Seed catalog so planning never goes empty when nearby search fails or returns nothing.
    seed_shops = retrieve_seed_candidates(
        city=expand_city,
        dietary_constraints=DietaryConstraints(
            excluded_shop_tags=plan_excluded_tags,
            appetite_light=appetite_light_mode,
            excluded_shop_names=excluded_shop_names,
        ),
        category_tags=must_have_tags,
        meal_slots=meal_slots_eff,
    )
    state["transit_audit"].append(
        _dj(
            "retrieval_service_seed",
            city=expand_city,
            catalog_stem=catalog_stem_for_city(expand_city),
            count=len(seed_shops),
            meal_slots=meal_slots_eff,
            category_tags=must_have_tags,
        )
    )

    for p in dynamic_pool[:60]:
        prof = _build_dynamic_shop_profile(p, region=str(p.get("region", "jp")))
        dynamic_shops.append(prof)
    _dyn_die_before = len(dynamic_shops)
    dynamic_shops = filter_shop_profiles_by_dietary_exclusions(dynamic_shops, plan_excluded_tags)
    if len(dynamic_shops) < _dyn_die_before:
        state["transit_audit"].append(
            _dj(
                "node_plan_dynamic_dietary_filtered",
                dropped=_dyn_die_before - len(dynamic_shops),
                kept=len(dynamic_shops),
            )
        )
    _dyn_excl_before = len(dynamic_shops)
    dynamic_shops = filter_shop_profiles_by_excluded_shop_names(dynamic_shops, excluded_shop_names)
    if len(dynamic_shops) < _dyn_excl_before:
        state["transit_audit"].append(
            _dj(
                "node_plan_dynamic_excluded_shop_names",
                dropped=_dyn_excl_before - len(dynamic_shops),
                kept=len(dynamic_shops),
                sample=list(excluded_shop_names)[:8],
            )
        )

    shops = seed_shops + dynamic_shops
    deduped_shops: list[ShopProfile] = []
    seen_shop_names: set[str] = set()
    for s in shops:
        if s.name in seen_shop_names:
            continue
        seen_shop_names.add(s.name)
        deduped_shops.append(s)
    added_dynamic = max(0, len(deduped_shops) - len(seed_shops))
    shops = deduped_shops

    if not dynamic_pool:
        state["transit_audit"].append(
            _dj(
                "dynamic_place_search_empty",
                candidates=0,
                detail="using seed-only pool until expansion",
            )
        )
    elif added_dynamic == 0:
        state["transit_audit"].append(
            _dj("dynamic_place_search_deduped_seed", new_dynamic_names=0)
        )

    # Multi-agent handoff: Researcher selected candidate shortlist.
    researcher_names = [str(x) for x in (state.get("researcher_candidate_names") or []) if str(x)]
    if researcher_names:
        name_set = set(researcher_names)
        shortlisted = [s for s in shops if s.name in name_set]
        if shortlisted:
            shops = shortlisted + [s for s in shops if s.name not in name_set]
            state["transit_audit"].append(
                _dj("researcher_candidates_applied", candidate_names=researcher_names, matched=len(shortlisted))
            )

    if must_have_tags:
        state["transit_audit"].append(
            _dj(
                "hard_tag_filter",
                explicit_tags=must_have_tags,
                note="softened_global; slot tags may apply per meal",
            )
        )
    shops_for_dynamic_planning = shops
    if must_have_tags:
        must_have_set = {t.lower() for t in must_have_tags}
        shops_for_dynamic_planning = [
            s for s in shops_for_dynamic_planning if any(str(tag).lower() in must_have_set for tag in s.tags)
        ]
        state["transit_audit"].append(
            _dj(
                "dynamic_plan_filtered_tags",
                tags=must_have_tags,
                kept=len(shops_for_dynamic_planning),
                total=len(shops),
            )
        )

    if appetite_light_mode:
        _ap_before = len(shops_for_dynamic_planning)
        shops_for_dynamic_planning = [
            s
            for s in shops_for_dynamic_planning
            if not (
                float(getattr(s, "portion_strictness", 0.5)) > 0.9
                and not bool(getattr(s, "has_small_portion", False))
            )
        ]
        state["transit_audit"].append(
            _dj(
                "appetite_light_portion_filter",
                kept=len(shops_for_dynamic_planning),
                before=_ap_before,
            )
        )

    # Probe clock isolated from synthesis start (also required before high-pressure slack scan).
    probe_now = copy.deepcopy(now)

    # Layer-3 runtime resilience: reserve probe outcomes for two tight high-score shops.
    high_pressure_pair: dict[str, str] = {}
    pressure_candidates: list[tuple[str, float, float]] = []  # (shop_name, urgency_score, trust)
    for s in shops_for_dynamic_planning:
        try:
            ch, cm = [int(x) for x in s.close_time.split(":", 1)]
            close_at = probe_now.replace(hour=ch, minute=cm, second=0, microsecond=0)
            last_call_at = close_at - timedelta(minutes=max(0, int(s.last_call_offset)))
            slack_min = (last_call_at - probe_now).total_seconds() / 60.0
        except Exception:
            continue
        if slack_min <= 0:
            continue
        if slack_min > 180:
            continue
        # Higher trust and shorter slack => higher priority for dual-path reservation.
        urgency = (300.0 - min(300.0, slack_min)) + (s.trust_score * 100.0)
        pressure_candidates.append((s.name, urgency, s.trust_score))
    pressure_candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
    if len(pressure_candidates) >= 2:
        a = pressure_candidates[0][0]
        b = pressure_candidates[1][0]
        high_pressure_pair[a] = b
        high_pressure_pair[b] = a
        state["transit_audit"].append(_dj("high_pressure_pair_reserved", shop_a=a, shop_b=b))

    review_samples: dict[str, dict] = {}
    review_index = {s.name: s for s in shops}
    for name, payload in review_samples.items():
        if name in review_index:
            setattr(review_index[name], "_review_texts", payload.get("reviews", []))

    shop_warning_badges: dict[str, str] = {}
    for shop in shops_for_dynamic_planning:
        # Demo override: user query can force probe keyword for testing.
        if "臨休" in state.get("query", "") and "燃えよ" in shop.name:
            sns_provider.fixtures[shop.sns_handle] = "本日臨休"
        plan = plan_shop_visit(
            shop=shop,
            current_time=probe_now,
            day_of_week="Sat",
            time_slot="lunch",
            from_loc="Kyoto Station" if "燃えよ" in shop.name else "Umeda",
            to_loc="Umeda" if "燃えよ" in shop.name else "Karasuma",
            travel_time_minutes=35 if "燃えよ" in shop.name else 18,
            dietary_preference=dietary_axis,
            sns_adapter=sns_provider,
            traffic_adapter=traffic_provider,
            candidate_shops=shops,
            use_driving_mode=prefers_driving_mode,
            allow_preorder_risk=advanced_mode,
        )
        if appetite_light_mode:
            _ps = float(getattr(shop, "portion_strictness", 0.5))
            if _ps > 0.9 and not bool(getattr(shop, "has_small_portion", False)):
                plan.preparation_note = (
                    f"{plan.preparation_note}；⚠ 此店份量較大且規定嚴格，若食量小建議改選有『小盛』標籤的店家"
                )
        if shop.name in high_pressure_pair:
            fallback = high_pressure_pair[shop.name]
            plan.preparation_note = (
                f"{plan.preparation_note}；此為高壓行程，若 {shop.name} 店排隊過長，Saga 引擎將自動切換至 {fallback} 備案"
            )
            state["transit_audit"].append(
                _dj("hybrid_saga_overbooking_note", shop=shop.name, fallback_shop=fallback)
            )

        review_data = review_samples.get(shop.name, {"google_negative_ratio": 0.3, "critic_score": 0.6, "reviews": []})
        review_analysis = ReviewAnalyzer.consensus_scoring(
            google_negative_ratio=review_data["google_negative_ratio"],
            critic_score=review_data["critic_score"],
            reviews=review_data["reviews"],
        )
        reliability_cutoff = _reliability_cutoff_for_region(shop.region)
        if review_analysis.final_reliability_score < reliability_cutoff:
            strongest_negative = review_analysis.negative_evidence[0] if review_analysis.negative_evidence else "（無可用負評樣本）"
            report += (
                f"| -- | {shop.name} | HIGH_BIAS_RISK | "
                f"Reliability={review_analysis.final_reliability_score}% "
                f"(cutoff={reliability_cutoff}, penalty={review_analysis.bias_penalty})；最真實負評：{strongest_negative} |\n"
            )
            state["transit_audit"].append(
                _dj(
                    "shop_review_high_bias_risk",
                    shop=shop.name,
                    reliability_score=review_analysis.final_reliability_score,
                )
            )
            continue

        if plan.outcome in ("FORCE_ABORT", "CONSTRAINT_CONFLICT", "PREORDER_EXPIRED_RISK"):
            backup = plan.backup_option or "N/A"
            report += (
                f"| -- | {shop.name} | {plan.semantic_status} | "
                f"{plan.preparation_note} Backup Option: {backup} |\n"
            )
            # Compensation-like behavior: fallback recommendation and continue.
            state["transit_audit"].append(
                _dj(
                    "plan_shop_visit_outcome",
                    shop=shop.name,
                    semantic_status=plan.semantic_status,
                    fallback=backup,
                )
            )
            continue
        if plan.outcome == "REROUTED_TRANSPORT":
            report += (
                f"| -- | {shop.name} | {plan.semantic_status} | {plan.preparation_note} |\n"
            )
            state["transit_audit"].append(
                _dj("plan_shop_visit_reroute", shop=shop.name, mode="taxi")
            )
            continue
        if "【紅色警告】" in plan.preparation_note:
            shop_warning_badges[shop.name] = plan.preparation_note

        for slot in plan.slots:
            report += (
                f"| {slot.start_at.strftime('%H:%M')} - {slot.end_at.strftime('%H:%M')} | "
                f"{slot.title} | {plan.semantic_status} | {plan.preparation_note} |\n"
            )
            probe_now = max(probe_now, slot.end_at + timedelta(minutes=15))

    # Decision optimization and itinerary synthesis
    user_pref = UserPreference(
        preferred_tags=["ramen", "scenic", "vegetarian"],
        avoid_tags=["nightlife"],
        dietary_preference="regular",
        max_wait_minutes=35,
        health_budget_limit=1.2,
        prefers_driving=prefers_driving_mode,
        max_total_minutes=115,
        max_budget_impact=0.72,
    )
    health_tracker = HealthBudgetTracker(spent=0.58)
    minefield = UserMinefield(
        blocked_shop_names={"Wildcard Izakaya"},
        blocked_tags={"too_salty", "賄賂送禮"},
    )
    query_upper = query_text.upper()
    is_taste_max = any(
        k in query_upper
        for k in ["TASTE_MAX", "好吃第一", "不計代價", "老饕", "美食狂熱"]
    )
    if right_now_mode:
        mode = OptimizationMode.RIGHT_NOW
    elif is_taste_max:
        mode = OptimizationMode.TASTE_MAX
    else:
        mode = OptimizationMode.BALANCED
    # Keep strategy routing deterministic for testability and reproducible demos.
    strategy = (
        SelectionStrategy.FOODIE_STRATEGY
        if "FOODIE_STRATEGY" in state.get("query", "")
        else SelectionStrategy.TOURIST_STRATEGY
    )
    base_weight_profile = WeightProfile.trust_first() if mode == OptimizationMode.RIGHT_NOW else (
        _learned_weight_profile if state.get("feedback_updates") else WeightProfile.trust_first()
    )
    preference_damping = bool(state.get("explicit_intent_preference_damping")) or _should_damp_preference_for_query(
        query_text
    )
    active_weight_profile = base_weight_profile
    if (
        preference_damping
        and mode != OptimizationMode.RIGHT_NOW
        and base_weight_profile.preference_bias > WeightProfile.trust_first().preference_bias + 1e-6
    ):
        active_weight_profile = _apply_runtime_weight_damping(base_weight_profile)
        state["transit_audit"].append(
            _dj(
                "preference_damping_runtime",
                trust_up=True,
                preference_down=True,
                reason="explicit query vs learned taste",
            )
        )
    ml_ready = _feedback_samples_seen >= _MIN_FEEDBACK_FOR_ML
    if mode == OptimizationMode.RIGHT_NOW:
        # RIGHT_NOW: keep only immediately reachable shops (default 30m travel budget).
        now_from = "Shinsaibashi" if ("心齋橋" in state.get("query", "") or "SHINSAIBASHI" in state.get("query", "").upper()) else "Kyoto Station"
        reachable: list[ShopProfile] = []
        for s in shops:
            if appetite_light_mode and float(getattr(s, "portion_strictness", 0.5)) > 0.9 and not getattr(
                s, "has_small_portion", False
            ):
                continue
            probe_plan = plan_shop_visit(
                shop=s,
                current_time=now,
                day_of_week=now.strftime("%a"),
                time_slot="dinner",
                from_loc=now_from,
                to_loc=s.neighborhood or s.name,
                travel_time_minutes=30,
                dietary_preference=dietary_axis,
                sns_adapter=sns_provider,
                traffic_adapter=traffic_provider,
                candidate_shops=shops,
                use_driving_mode=prefers_driving_mode,
            )
            if probe_plan.outcome in {"SUCCESS", "REROUTED_TRANSPORT"}:
                reachable.append(s)
        shops = reachable
        state["transit_audit"].append(
            _dj("right_now_mode_filtered", reachable=len(shops), origin_anchor=now_from)
        )

    # Phase 1: Heuristic Filter (tag-aware, fast pre-ranking to top-15 candidates)
    state["transit_audit"].append(_dj("hybrid_phase1_heuristic_filter", phase="start"))
    nearby_counts = {
        "燃えよ麺助": 1,
        "Harbs 大丸京都": 5,
        "一蘭 京都河原町": 6,
        "喫茶ソワレ": 2,
        "松籟庵": 1,
        "Wildcard Izakaya": 4,
    }
    heuristic_scored: list[RankedShop] = []
    must_have_set = {t.lower() for t in must_have_tags}
    for s in shops:
        if must_have_set and not any(str(t).lower() in must_have_set for t in s.tags):
            continue
        if appetite_light_mode and float(getattr(s, "portion_strictness", 0.5)) > 0.9 and not getattr(
            s, "has_small_portion", False
        ):
            continue
        alpha = DensityScanner.scan_isolation_factor(nearby_counts.get(s.name, 3))
        score, pref_match, _ = ScoringEngine.score_with_isolation(
            s,
            user_pref,
            active_weight_profile,
            isolation_factor=alpha,
            isolation_threshold=0.7,
        )
        penalty = max(0.0, float(_feedback_penalties.get(s.name, 0.0)))
        score = max(0.0, score - penalty)
        if appetite_light_mode and getattr(s, "has_small_portion", False):
            score *= 1.15
        heuristic_scored.append(RankedShop(shop=s, final_score=float(score), preference_match_score=float(pref_match)))
    heuristic_scored.sort(key=lambda x: x.final_score, reverse=True)
    requested_slots = meal_slots_eff
    requested_meal_count = _requested_meal_count(query_text)
    slot_required_tags = _slot_level_required_tags(query_text)
    if slot_required_tags:
        state["transit_audit"].append(
            _dj(
                "slot_specific_tags",
                slots={k: sorted(v) for k, v in slot_required_tags.items()},
            )
        )
    if requested_slots:
        state["transit_audit"].append(
            _dj("meal_slot_partitioning", slots=requested_slots)
        )
    phase_cap = max(22, len(requested_slots) + 18) if requested_slots else 22
    phase1_candidates = _inject_slot_anchor_rankeds_into_phase1(
        core_head=heuristic_scored[:15],
        full_scores=heuristic_scored,
        seed_profiles=seed_shops,
        slots=requested_slots,
        query=query_text,
        cap=phase_cap,
    )
    phase1_shops = [x.shop for x in phase1_candidates]
    state["transit_audit"].append(
        _dj(
            "hybrid_phase1_heuristic_filter",
            phase="done",
            kept=len(phase1_candidates),
            slot_injection=max(0, len(phase1_candidates) - min(15, len(heuristic_scored))),
        )
    )

    ranked, rejected_list = RankingEngine.generate_top_picks(
        shops=phase1_shops,
        preference=user_pref,
        minefield=minefield,
        weight_profile=active_weight_profile,
        nearby_counts=nearby_counts,
        isolation_threshold=0.7,
        mode=mode,
        health_tracker=health_tracker,
        selection_strategy=strategy,
        learner=_pref_learner if ml_ready else None,
        skip_semantic_mines=(mode == OptimizationMode.RIGHT_NOW),
        shop_penalties=_feedback_penalties,
        must_have_tags=must_have_tags,
        taste_max_blacklist=_taste_max_blacklist,
        appetite_light_mode=appetite_light_mode,
    )
    if mode == OptimizationMode.TASTE_MAX:
        authority_hit_count = sum(
            1
            for r in ranked
            if (
                r.shop.authority_data.michelin_star > 0
                or "百名店" in r.shop.authority_data.tablelog_medal
                or len(r.shop.authority_data.chef_lineage) > 0
            )
        )
        state["transit_audit"].append(
            _dj(
                "taste_authority_scan",
                ranked=len(ranked),
                authority_hits=authority_hit_count,
            )
        )
    if not ml_ready:
        state["transit_audit"].append(
            _dj(
                "feedback_model_pending",
                collected_samples=_feedback_samples_seen,
                required_samples=_MIN_FEEDBACK_FOR_ML,
            )
        )
    if appetite_light_mode:
        state["transit_audit"].append(_dj("appetite_light_mode", enabled=True))
    feedback_applied = bool(_feedback_penalties) or bool(state.get("feedback_updates"))
    state["transit_audit"].append(
        _dj(
            "plan_mode_summary",
            mode=mode.value,
            feedback_applied=feedback_applied,
            meal_slot_optimized=bool(requested_slots),
        )
    )
    # Phase 2: DP Solver on top-15 (calendar-pin GraphBuilder wrapper: single excursion day vs per-node align drift)
    with _pinned_travel_dp_calendar_and_solver():
        state["transit_audit"].append(
            _dj(
                "dp_graph_itinerary_calendar_pin_armed",
                itinerary_date=str(synth_start_time.date()),
            )
        )
        state["transit_audit"].append(_dj("hybrid_phase2_dp_solver", phase="start"))
        graph = GraphBuilder.build_graph(
            ranked=phase1_candidates,
            traffic=traffic_provider,
            start_time=synth_start_time,
            meal_slots=requested_slots,
            mode=mode,
            requested_meal_count=requested_meal_count,
            slot_required_tags=slot_required_tags if slot_required_tags else None,
            excluded_shop_tags=plan_excluded_tags,
        )
        _append_graph_physical_transition_audit(
            graph=graph,
            ranked=phase1_candidates,
            traffic=traffic_provider,
            transit_audit=state["transit_audit"],
            mode=mode,
        )
        desired_len = max(1, requested_meal_count or len(requested_slots or phase1_candidates[:3]))
        _k_meal_slots_norm = ItinerarySynthesizer._normalize_slot_sequence(requested_slots) if requested_slots else []
        k_paths = ItinerarySynthesizer.find_k_optimal_paths(
            graph=graph,
            required_length=desired_len,
            must_have_tags=explicit_category_tags,
            k=5,
            meal_slots=_k_meal_slots_norm if _k_meal_slots_norm else None,
            excluded_shop_tags=plan_excluded_tags,
        )
        state["transit_audit"].append(
            _dj("hybrid_phase2_dp_solver", phase="done", paths=len(k_paths))
        )

        # Phase 3: Saga Commitment (SNS probe + flight lock), fallback to next-best path on failure.
        state["transit_audit"].append(_dj("hybrid_phase3_saga_commitment", phase="start"))
        ranked_by_name = {r.shop.name: r for r in phase1_candidates}
        selected_ranked_path: list[RankedShop] = []
        phase3_used_fallback = False
        risk_keywords = ("火山", "臨休", "休業", "完売", "sold out")
        for idx, path in enumerate(k_paths, start=1):
            path_ordered = sorted(path, key=lambda n: (n.slot_index, n.shop_name))
            path_names = [n.shop_name for n in path_ordered]
            phase3_failed = False
            for name in path_names:
                r = ranked_by_name.get(name)
                if r is None:
                    continue
                signal = sns_provider.check_store_status(r.shop.sns_handle).lower()
                if any(k in signal for k in risk_keywords):
                    state["transit_audit"].append(
                        _dj(
                            "hybrid_phase3_fail",
                            path_index=idx,
                            shop=name,
                            reason="sns_risk",
                        )
                    )
                    phase3_failed = True
                    break
            if phase3_failed:
                continue
            selected_ranked_path = [ranked_by_name[n] for n in path_names if n in ranked_by_name]
            state["transit_audit"].append(
                _dj("hybrid_phase3_commit", path_index=idx, shops=path_names)
            )
            break
        if not selected_ranked_path:
            phase3_used_fallback = True
            fallback_pool = ranked
            if plan_excluded_tags:
                fb = [
                    r
                    for r in ranked
                    if not ItinerarySynthesizer.shop_has_excluded_tag(r.shop, plan_excluded_tags)
                ]
                if not fb:
                    fb = [
                        rc
                        for rc in phase1_candidates
                        if not ItinerarySynthesizer.shop_has_excluded_tag(rc.shop, plan_excluded_tags)
                    ]
                if fb:
                    fallback_pool = fb
                else:
                    state["transit_audit"].append(
                        _dj(
                            "phase3_fallback_empty_after_tag_exclusions",
                            excluded_count=len(plan_excluded_tags),
                            ranked_len=len(ranked),
                        )
                    )
                    fallback_pool = []
            if fallback_pool:
                take_n = max(1, min(3, len(fallback_pool)))
                selected_ranked_path = fallback_pool[:take_n]
            else:
                selected_ranked_path = []
            state["transit_audit"].append(
                _dj(
                    "hybrid_phase3_fallback",
                    detail="no committed path; using heuristic top picks",
                )
            )

        ranked = selected_ranked_path
        _binding_slots_norm = ItinerarySynthesizer._normalize_slot_sequence(
            requested_slots or []
        )
        respect_slot_order = (
            (not phase3_used_fallback)
            and len(_binding_slots_norm) > 0
            and len(ranked) == len(_binding_slots_norm)
        )
        if respect_slot_order:
            state["transit_audit"].append(
                _dj(
                    "synth_respects_dp_slot_order",
                    slots=_binding_slots_norm,
                    shops=[r.shop.name for r in ranked],
                )
            )
        synth_mode = OptimizationMode.BALANCED if mode == OptimizationMode.RIGHT_NOW else mode
        synthesized = ItinerarySynthesizer.synthesize(
            ranked,
            traffic_provider,
            user_pref,
            start_time=synth_start_time,
            sns_adapter=sns_provider,
            isolation_threshold=0.7,
            mode=synth_mode,
            meal_slots=requested_slots,
            global_end_time=global_end_dt,
            requested_meal_count=requested_meal_count,
            explicit_required_tags=explicit_category_tags,
            slot_required_tags=slot_required_tags if slot_required_tags else None,
            appetite_light_mode=appetite_light_mode,
            respect_slot_order=respect_slot_order,
            excluded_shop_tags=plan_excluded_tags,
        )
    boundary_skips = [w for w in synthesized.warnings if w.startswith("OPERATING_BOUNDARY_SKIP")]
    for skip_msg in boundary_skips:
        state["transit_audit"].append(
            _dj("synthesis_warning_forwarded", warning=skip_msg)
        )
    if synthesized.rollback_triggered:
        state["transit_audit"].append(_dj("early_interception_rollback_triggered"))

    # Health_Check before commitment: if over budget, rollback to healthier backup.
    if ranked:
        top_shop = ranked[0].shop
        projected_health = health_tracker.spent + ScoringEngine.optimized_health_impact(top_shop)
        if projected_health > user_pref.health_budget_limit:
            fallback = choose_health_backup([r.shop for r in ranked], top_shop)
            if fallback is not None:
                state["transit_audit"].append(
                    _dj(
                        "health_check_rollback",
                        from_shop=top_shop.name,
                        to_shop=fallback.name,
                    )
                )
                report += (
                    f"\n\n> Health_Check: `{top_shop.name}` 導致健康預算超標 "
                    f"({projected_health:.2f} > {user_pref.health_budget_limit:.2f})，"
                    f"已觸發 Saga 回滾並切換至 `{fallback.name}`。"
                )
            else:
                state["transit_audit"].append(
                    _dj("health_check_rollback", detail="no_fallback_found")
                )
                report += (
                    f"\n\n> Health_Check: `{top_shop.name}` 導致健康預算超標，"
                    "已觸發 Saga 回滾，但無可用健康備案。"
                )

    report += f"\n\n## Section 1: 推薦排行榜 (Top Picks) — Mode: `{mode.value}`\n"
    report += "| Rank | Shop | FinalScore | Note |\n|---|---|---:|---|\n"
    for idx, item in enumerate(ranked, start=1):
        note = "Wildcard" if item.is_wildcard else "Core Pick"
        if item.rank_note:
            note = f"{note}; {item.rank_note}"
        alpha = getattr(item.shop, "_isolation_factor", 0.0)
        if alpha >= 0.7:
            note = f"{note}; IsolationFactor={alpha}"
        report += f"| {idx} | {item.shop.name} | {item.final_score:.2f} | {note} |\n"

    report += "\n## Section 2: 韌性行程表 (Resilient Schedule)\n"
    report += "| Time | Node | Note |\n|---|---|---|\n"
    for n in synthesized.nodes:
        report += (
            f"| {n.start_at.strftime('%H:%M')} - {n.end_at.strftime('%H:%M')} | "
            f"{n.title} | {n.note} |\n"
        )
    for warning in synthesized.warnings:
        report += f"\n> {warning}\n"
    if synthesized.backup_nodes:
        report += f"\nBackupNode candidates: {', '.join(synthesized.backup_nodes)}\n"
    report += "\n### Solver Diagnostics (Debug Only)\n"
    transit_audit = state.get("transit_audit", [])
    if not synthesized.solver_audit_log and not synthesized.graph_debug_traces and not transit_audit:
        report += "- No solver diagnostics.\n"
    else:
        for audit in synthesized.solver_audit_log:
            report += f"- [DP_AUDIT] `{audit}`\n"
        for trace in synthesized.graph_debug_traces:
            report += f"- [GRAPH_TRACE] `{trace}`\n"
            explained = False
            try:
                g = json.loads(trace)
                if isinstance(g, dict) and g.get("event") == "graph_rejected_edge":
                    report += (
                        f"  - {g.get('from_shop', '')} 之後無法接 {g.get('to_shop', '')}，"
                        f"因為 {g.get('detail', '')}\n"
                    )
                    explained = True
            except json.JSONDecodeError:
                pass
            if not explained:
                m = re.search(r"\[REJECTED_EDGE\] 從 (.+?) 到 (.+?) 失敗：(.+)。", trace)
                if m:
                    from_shop, to_shop, reason = m.groups()
                    report += f"  - {from_shop} 之後無法接 {to_shop}，因為 {reason}\n"
        for audit in transit_audit:
            report += f"- [FLOW_AUDIT] `{audit}`\n"

    report += "\n## Section 3: 避雷報告 (Minefield Check)\n"
    if not rejected_list:
        report += "- No hard-drop entries.\n"
    else:
        report += "| Shop | EstimatedScore | FilterReason |\n|---|---:|---|\n"
        for rj in rejected_list:
            report += f"| {rj.shop_name} | {rj.estimated_score:.2f} | {rj.reason} |\n"

    report += "\n## MinefieldAnalysis\n"
    famous_rejected = [rj for rj in rejected_list if any(s.name == rj.shop_name and s.is_famous for s in shops)]
    if not famous_rejected:
        report += "- 無名店被語義地雷過濾。\n"
    else:
        for rj in famous_rejected:
            report += f"- {rj.shop_name} 已過濾：{rj.reason}\n"

    report += "\n---\n### Conversational rollback\n"
    report += "Use graph thread checkpoints for rollback/time-travel.\n"
    report += f"Thread ID: `{state.get('agent_run_id','')}`\n"
    ui_cards: list[dict] = []
    for item in ranked[:5]:
        shop = item.shop
        if getattr(item, "insider_pick", False):
            why = "【老饕私藏】此店名氣較低，但味覺信號純粹，避開了權威獎項的行銷噪音"
        else:
            why = item.rank_note or (item.top_3_reasons[0] if item.top_3_reasons else "整體風險較低且口味匹配。")
        to_go = "建議搭乘大眾運輸前往。"
        if prefers_driving_mode:
            to_go = "建議自駕或計程車，保留交通緩衝。"
        reserve_hint = "可直接現場候位"
        if shop.booking_type == BookingType.PHONE:
            reserve_hint = f"建議電話預約 {shop.booking_phone}".strip()
        elif shop.booking_type == BookingType.WEB:
            reserve_hint = "建議先透過官網或平台訂位"
        ui_cards.append(
            {
                "shop_name": shop.name,
                "address_hint": shop.neighborhood or "未提供",
                "why_selected": why,
                "how_to_go": to_go,
                "reservation_hint": reserve_hint,
                "rank_note": item.rank_note,
                "insider_pick": getattr(item, "insider_pick", False),
                "warning_badge": "PREORDER_WARNING" if shop.name in shop_warning_badges else "",
                "warning_text": shop_warning_badges.get(shop.name, ""),
                "lat": shop.latitude,
                "lng": shop.longitude,
            }
        )
    state["ui_cards"] = ui_cards
    state["final_itinerary"] = report
    return state


@traced
def node_collect_feedback(state: AgentState) -> AgentState:
    global _learned_weight_profile, _feedback_samples_seen, _feedback_penalties, _taste_max_blacklist
    if state.get("error"):
        return state
    query = state.get("query", "")
    # 後續 plan 可用：當輪 query 明示甜點／串燒等時，對歷史偏好做 runtime damping（不依賴 feedback:）
    state["explicit_intent_preference_damping"] = _should_damp_preference_for_query(query)

    marker = "feedback:"
    if marker not in query.lower():
        return state

    fame_complaint = _feedback_complains_fame_unreliable(query)

    pref = UserPreference(
        preferred_tags=["ramen", "scenic", "vegetarian"],
        avoid_tags=["nightlife"],
        dietary_preference="regular",
        max_wait_minutes=35,
        health_budget_limit=1.2,
        max_total_minutes=115,
        max_budget_impact=0.72,
    )
    shops = _build_shop_catalog()
    by_name = {s.name: s for s in shops}

    payload = query.lower().split(marker, 1)[1]
    raw_pairs = [x.strip() for x in payload.split(",") if "=" in x]
    samples: list[tuple[ShopProfile, int]] = []
    updates: list[str] = []
    for raw in raw_pairs:
        left, right = raw.split("=", 1)
        shop_name = left.strip()
        try:
            stars = int(right.strip())
        except ValueError:
            continue
        resolved = next((k for k in by_name if k.lower() == shop_name.lower()), None)
        if not resolved:
            continue
        clamped = max(1, min(5, stars))
        samples.append((by_name[resolved], clamped))
        updates.append(f"{resolved}={clamped}星")
        current_penalty = float(_feedback_penalties.get(resolved, 0.0))
        if clamped <= 2:
            # Strongly down-rank shops with explicit low-star feedback.
            delta = 18.0 if clamped == 1 else 10.0
            _feedback_penalties[resolved] = min(45.0, current_penalty + delta)
            if clamped == 1 and "一蘭" in resolved:
                _taste_max_blacklist.add(resolved)
        elif clamped >= 4 and current_penalty > 0:
            # Positive feedback gradually releases penalty lock.
            _feedback_penalties[resolved] = max(0.0, current_penalty - 6.0)
            if resolved in _taste_max_blacklist:
                _taste_max_blacklist.remove(resolved)

    if not samples and not fame_complaint:
        return state

    learning_mode = ""
    if samples:
        _feedback_samples_seen += len(samples)
        if _feedback_samples_seen >= _MIN_FEEDBACK_FOR_ML:
            _pref_learner.partial_fit(samples, pref)
            _learned_weight_profile = _pref_learner.to_weight_profile()
            learning_mode = "ml_elasticnet"
            if state.get("explicit_intent_preference_damping"):
                _learned_weight_profile = ScoringEngine.normalize_weights(
                    WeightProfile(
                        trust_bias=min(1.0, _learned_weight_profile.trust_bias + 0.06),
                        preference_bias=min(_learned_weight_profile.preference_bias, 0.48),
                        logistics_bias=max(0.05, _learned_weight_profile.logistics_bias),
                    )
                )
                updates.append("explicit_intent:ml_pref_cap")
        else:
            # Lightweight fallback before enough samples for stable ML training.
            avg_stars = sum(stars for _, stars in samples) / len(samples)
            delta = (avg_stars - 3.0) / 2.0 * 0.05
            pref_delta = delta
            if state.get("explicit_intent_preference_damping"):
                # 使用者本輪已明示新菜系時，弱化偏好權重的更新幅度（避免過度擬合上一輪口味）
                pref_delta *= 0.35
                updates.append("explicit_intent:pref_delta_scaled")
            updated = WeightProfile(
                trust_bias=max(0.05, _learned_weight_profile.trust_bias - pref_delta / 2.0),
                preference_bias=max(0.05, _learned_weight_profile.preference_bias + pref_delta),
                logistics_bias=max(0.05, _learned_weight_profile.logistics_bias - pref_delta / 2.0),
            )
            _learned_weight_profile = ScoringEngine.normalize_weights(updated)
            learning_mode = "lightweight_ewma"

    if fame_complaint:
        _learned_weight_profile = ScoringEngine.normalize_weights(
            WeightProfile(trust_bias=0.3, preference_bias=0.7, logistics_bias=0.0)
        )
        updates.append("FAME_MISMATCH_OVERRIDE:trust=0.3,preference=0.7")
        if not learning_mode:
            learning_mode = "fame_mistrust_override"

    state["feedback_updates"] = updates
    state["learned_weight_profile"] = {
        "trust_bias": round(_learned_weight_profile.trust_bias, 4),
        "preference_bias": round(_learned_weight_profile.preference_bias, 4),
        "logistics_bias": round(_learned_weight_profile.logistics_bias, 4),
    }
    state["transit_audit"].append(
        _dj(
            "feedback_learning_updated",
            mode=learning_mode,
            samples_seen=_feedback_samples_seen,
            weights=state["learned_weight_profile"],
            penalties=dict(_feedback_penalties),
            taste_max_blacklist=sorted(_taste_max_blacklist),
            updates=updates,
        )
    )
    if fame_complaint:
        state["transit_audit"].append(
            _dj(
                "fame_mistrust_weights",
                trust_bias=0.3,
                preference_bias=0.7,
                note="flavor_vector alignment",
            )
        )
    _save_learning_state()
    return state


@traced
def node_synthesizer(state: AgentState, config: Optional[RunnableConfig] = None) -> AgentState:
    """On-demand transcript mediator — runs only when the user pauses.

    Reads retrieval_history + critique_history, calls SynthesizerAgent (Gemini),
    writes SynthesisReport into synthesis_history.  Never triggers automatically.
    """
    if state.get("error"):
        return state
    print(_dj("debug_print", node="node_synthesizer", message="SynthesizerAgent starting"))
    try:
        agent = SynthesizerAgent(llm_router=_svc_llm_router(state))
        report = agent.run(state)
    except Exception as exc:
        _attach_node_error(state, "synthesizer", exc)
        state.setdefault("transit_audit", []).append(_dj("synthesizer_failed", reason=str(exc)[:800]))
        return state

    state["synthesis_history"] = list(state.get("synthesis_history") or []) + [report.as_dict()]
    state["transit_audit"].append(
        _dj(
            "synthesis_complete",
            retrieval_rounds=report.retrieval_rounds,
            critique_rounds=report.critique_rounds,
            discussion_freshness=report.discussion_freshness,
            consensus_count=len(report.consensus),
            unresolved_count=len(report.unresolved),
            frontier_count=len(report.frontier),
        )
    )
    return state


def build_graph(*, interrupt_after_nodes: list[str] | None = None, checkpointer: object | None = None):
    """Build the main LangGraph agent: Retriever → Critic loop → Plan.

    Parameters
    ----------
    interrupt_after_nodes:
        If ``checkpointer`` is set, LangGraph ``interrupt_after`` is **only** applied when this
        argument is explicitly provided (including ``[]`` for no interrupts).
        Pass ``None`` (default): compile with ``interrupt_after=[]`` so streaming runs finish
        without an automatic pause after ``retriever`` / ``critic``.

    checkpointer:
        Optional LangGraph checkpointer (e.g. ``AsyncSqliteSaver`` wired in FastAPI ``lifespan``).
        Async savers must be driven only from async graph APIs (``aget_state``, ``astream``, ``aupdate_state``, …).
        Tests call ``build_graph()`` with no arguments so graphs compile without persistence.

    Graph topology
    --------------
    route_intent → clarify_constraint → retriever → researcher → critic ⟲ (request_more → researcher, max-iter guard)
                                  ↓ satisfied / deadlock
                             collect_feedback → plan → END  (``plan`` appends LangGraph checkpoint id to ``turn_checkpoints``)
    If ``clarify_constraint`` needs a strict/loose answer, the graph routes to END until the user replies (A/B).

    synthesizer → END             (standalone; same bookkeeping when invoked in-graph)

    Note: node_researcher is still available but the critic verdict drives the loop.
    Flight-booking (node_flight_search) and Saga-reservation (node_audit) are
    intentionally excluded — see the ADR in README.md.
    """
    g = StateGraph(AgentState)
    g.add_node("route_intent", node_route_intent)
    g.add_node("clarify_constraint", node_clarify_constraint)
    g.add_node("retriever", node_retriever)
    g.add_node("researcher", node_researcher)
    g.add_node("critic", node_critic)
    g.add_node("synthesizer", node_synthesizer)
    g.add_node("collect_feedback", node_collect_feedback)
    g.add_node("plan", node_plan)
    g.set_entry_point("route_intent")

    def _after_clarify_constraint(state: AgentState) -> str:
        if state.get("awaiting_dietary_clarification"):
            return "end"
        return "retriever"

    g.add_edge("route_intent", "clarify_constraint")
    g.add_conditional_edges(
        "clarify_constraint",
        _after_clarify_constraint,
        {"end": END, "retriever": "retriever"},
    )
    g.add_edge("retriever", "researcher")
    g.add_edge("researcher", "critic")

    def _after_critic(state: AgentState) -> str:
        if state.get("error"):
            if (state.get("intent") or {}).get("mode") == "right_now":
                return "plan"
            return "collect_feedback"
        # CriticAgent verdict drives the loop
        verdict = (state.get("critique_history") or [{}])[-1].get("verdict", "")
        rejected = bool(state.get("auditor_rejected"))
        iteration = int(state.get("research_iteration", 0))

        if verdict == "satisfied":
            pass  # fall through to plan/collect
        elif (rejected or verdict == "request_more") and iteration < 3:
            return "researcher"
        # satisfied / deadlock / max-iterations → proceed
        if (state.get("intent") or {}).get("mode") == "right_now":
            return "plan"
        return "collect_feedback"

    g.add_conditional_edges(
        "critic",
        _after_critic,
        {"researcher": "researcher", "plan": "plan", "collect_feedback": "collect_feedback"},
    )
    g.add_edge("collect_feedback", "plan")
    g.add_edge("plan", END)

    # Synthesizer is a standalone on-demand node: synthesizer → END
    g.add_edge("synthesizer", END)

    if checkpointer is not None:
        _interrupt: list[str] = (
            list(interrupt_after_nodes) if interrupt_after_nodes is not None else []
        )
        return g.compile(checkpointer=checkpointer, interrupt_after=_interrupt)
    return g.compile()


if __name__ == "__main__":
    async def _cli() -> AgentState:
        initial_state_local: AgentState = make_initial_state("Book TPE to SFO via NRT")
        return await build_graph().ainvoke(initial_state_local)

    result = asyncio.run(_cli())
    print(_dj("cli_demo_done", itinerary_chars=len(result.get("final_itinerary", "") or "")))
    print(result["final_itinerary"])
