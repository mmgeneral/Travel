from __future__ import annotations
import numpy as np
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
from config import C_INT
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
    contender_set,
    rerank_by_posterior,
    freeze_phase_b_turn_context,
    phase_b_rerank,
    generate_cross_block_questions,
    compute_evoi_for_questions,
    evaluate_gate,
)

from intent_parser import intent_from_snapshot_dict as _intent_from_snapshot_dict
from intent_parser import parse_intent as _parse_intent
from tracing import trace_agent_stage
from llm_router import TaskType, LLMRouter as _LLMRouter
from agents.retriever import RetrieverAgent
from agents.critic import CriticAgent
from dp_solver import (
    _haversine_km,
    node_critic,
    node_auditor,
    node_audit,
    agent_dp_find_optimal_path_no_shop_repeat,
    _dp_graph_builder_with_itinerary_day_pin,
)
from checkpoint_entry import StandardCheckpointEntry, entries_from_state, entries_to_state
from agents.synthesizer import SynthesizerAgent, SynthesisReport
from observability import traced
from dietary_utils import (
    _user_negates_food_category_in_query,
    _constraint_string_to_dietary_keys,
    _dietary_keys_from_query_and_intent_signals,
    _build_plan_excluded_shop_tags,
    _canonical_dietary_hints_list,
    _ambiguous_dietary_hint_for_clarification,
    _DIETARY_CLARIFICATION_QUESTIONS,
    _parse_dietary_clarification_reply,
    _strip_dietary_hint_key_from_intent,
    _consume_pending_dietary_clarification_answer,
    _llm_extract_city_from_choice,
    _apply_city_from_clarification,
    _consume_pending_intent_clarification,
)
from query_utils import (
    TimeRange,
    _extract_user_time_window,
    _extract_time_window,
    _is_ramen_intent,
    _has_strong_ramen_intent,
    _extract_explicit_category_tags,
    _direct_food_category_mentions,
    _softened_explicit_category_tags,
    _plan_global_explicit_tags,
    _should_damp_preference_for_query,
    _apply_runtime_weight_damping,
    _slot_level_required_tags,
    _extract_search_category_keywords,
    _seed_shop_tag_bag,
    _seed_single_covers_slot,
    _seed_covers_meal_slot,
    _researcher_semantic_overlap,
    _researcher_clock_minutes,
    _researcher_slot_required_gate,
    _researcher_time_fit_slot,
    _researcher_seed_eligible_for_slot,
    _researcher_shop_eligible_any_tier,
    _researcher_score_tuple,
    _researcher_best_seed_for_slot,
    _researcher_slot_anchor_names,
    _researcher_finalize_candidate_names,
    _plan_dynamic_place_queries,
    _is_appetite_light_intent,
    _feedback_complains_fame_unreliable,
    _SLOT_FORCED_PLACES_QUERY,
    _requested_meal_count,
    _requested_meal_slots,
    _effective_plan_meal_slots,
)

from geo_utils import (
    _locale_token_to_city_region,
    _infer_city_region_from_coords,
    _extract_city_from_query,
    _is_flight_booking_intent,
    _fallback_broad_geo_queries,
)
from shop_profile_utils import (
    _clamp_hhmm_token,
    _infer_close_time,
    _infer_open_time,
    _resolve_dynamic_open_time,
    _is_dynamic_time_unknown,
    _build_dynamic_shop_profile,
    _dynamic_pool_row_from_place,
    _reliability_cutoff_for_region,
    _make_cache_key,
    _schedule_slots,
)

_RAW_GRAPHBUILDER_BUILD = GraphBuilder.build_graph

class AgentState(TypedDict):
    # Phase A state
    phase_a_posterior_mu: list[float]
    phase_a_posterior_sigma: list[list[float]]
    phase_a_evidence_log: list[dict]
    phase_a_trip_feature_scaling: dict | None
    asked_this_turn: bool
    # Phase B state (frozen B-turn Context)
    phase_b_turn_context: dict | None
    phase_b_contender_size: int | None
    phase_b_contender_meta: dict | None
    phase_b_gate_skipped: bool
    # Phase C state
    phase_c_event_xe: list[float] | None
    phase_c_evoi_values: list[float]
    phase_c_gate: dict | None
    phase_c_attribution_already_given: bool
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
    turn_checkpoints: list[dict]
    dietary_clarification_resolved: dict[str, str]
    pending_dietary_clarification: dict[str, Any] | None
    awaiting_dietary_clarification: bool
    clarification_broadcast: dict[str, Any] | None
    #: True when intent gate refused to run retrieval; user must answer (A)–(D) or clarify geography.
    awaiting_intent_clarification: bool
    #: Set by a failing node → downstream nodes noop; orchestrator maps to client errors.
    error: dict[str, Any] | None
    #: Injected deps (``llm_router``, ``openai_chat_client``, ``flight_service``); empty {} uses module defaults.
    runtime_services: dict[str, Any]
    #: Count of how many times the critic node has been executed in the current turn.
    critic_retry_count: int
    #: Global schedule (slot -> list of tags) passed from the frontend for collision detection.
    global_schedule: dict[str, list[str]] | None
    #: Stable UUID-keyed slots; each entry: {slot_id, meal_type, shop_name, locked}
    itinerary_slots: list[dict]
    #: key=(meal_type, frozenset(tags)) serialized as str, value=list[str]
    candidate_cache: dict[str, Any]
    #: Conflict found during revision transport check; None if no conflict.
    conflict: dict | None
    #: key=shop_name, value=[checkpoint_id, ...] 按時間排序
    shop_checkpoint_index: dict[str, list[str]]


class AgentStateModel(BaseModel):
    phase_a_posterior_mu: list[float] = Field(default_factory=lambda: [0.0] * 6)
    phase_a_posterior_sigma: list[list[float]] = Field(
        default_factory=lambda: [ [1.0 if i == j else 0.0 for j in range(6)] for i in range(6) ]
    )
    phase_a_evidence_log: list[dict] = Field(default_factory=list)
    phase_a_trip_feature_scaling: dict | None = None
    asked_this_turn: bool = False
    phase_b_turn_context: dict | None = None
    phase_b_contender_size: int | None = None
    phase_b_contender_meta: dict | None = None
    phase_b_gate_skipped: bool = False
    phase_c_event_xe: list[float] | None = None
    phase_c_evoi_values: list[float] = Field(default_factory=list)
    phase_c_gate: dict | None = None
    phase_c_attribution_already_given: bool = False
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
    turn_checkpoints: list[dict] = Field(default_factory=list)
    plan_excluded_shop_tags: list[str] = Field(default_factory=list)
    dietary_clarification_resolved: dict[str, str] = Field(default_factory=dict)
    pending_dietary_clarification: dict[str, Any] | None = None
    awaiting_dietary_clarification: bool = False
    clarification_broadcast: dict[str, Any] | None = None
    awaiting_intent_clarification: bool = False
    error: dict[str, Any] | None = None
    runtime_services: dict[str, Any] = Field(default_factory=dict)
    critic_retry_count: int = 0
    global_schedule: dict[str, list[str]] | None = None
    itinerary_slots: list[dict] = Field(default_factory=list)
    candidate_cache: dict[str, Any] = Field(default_factory=dict)
    conflict: dict | None = None
    shop_checkpoint_index: dict[str, list[str]] = Field(default_factory=dict)


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
    global_schedule: dict[str, list[str]] | None = None,
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
        global_schedule=global_schedule,
    )
    return model.model_dump(mode="json")


def invalidate_candidate_cache(state: dict) -> dict:
    """Call this when shop catalog is updated."""
    state["candidate_cache"] = {}
    return state


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
            temperature=0.0,
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


def _get_shop_catalog(state: AgentState) -> dict:
    """
    Build a catalog dict keyed by shop name with latitude/longitude fields.
    Reuses the same pattern as _node_plan_core.
    """
    seed_profiles = _researcher_seed_catalog(state)
    _intent = state.get("intent") or {}
    region = _intent.get("region") or "jp"
    dynamic_profiles = [_build_dynamic_shop_profile(p, region=region) for p in (state.get("dynamic_shop_pool") or [])]
    catalog = {}
    for sp in seed_profiles + dynamic_profiles:
        catalog[sp.name] = sp
    return catalog


def _conflict_check_travel(
    slots: list[dict],
    target_slot_id: str,
    shop_catalog: dict,
) -> dict | None:
    """
    Check if the target slot's travel time to/from adjacent slots is feasible.
    Returns None if OK, or a conflict dict if not feasible.

    conflict dict:
    {
        "conflict_type": "transport",
        "slot_id": str,
        "prev_shop": str | None,
        "next_shop": str | None,
        "feasible_window": {"earliest": "HH:MM", "latest": "HH:MM"} | None,
        "message": str,
    }
    """
    SPEED_MIN_PER_KM = 2
    MIN_TRAVEL = 5

    def _get_coord(shop_obj, lat_key="latitude", lng_key="longitude"):
        if isinstance(shop_obj, dict):
            return shop_obj.get(lat_key), shop_obj.get(lng_key)
        return getattr(shop_obj, lat_key, None), getattr(shop_obj, lng_key, None)

    def travel_min(shop_a: str, shop_b: str) -> int:
        a = shop_catalog.get(shop_a, {})
        b = shop_catalog.get(shop_b, {})
        lat_a, lon_a = _get_coord(a)
        lat_b, lon_b = _get_coord(b)
        if None in (lat_a, lon_a, lat_b, lon_b):
            return MIN_TRAVEL
        km = _haversine_km(lat_a, lon_a, lat_b, lon_b)
        return max(MIN_TRAVEL, int(km * SPEED_MIN_PER_KM))

    # 找 target slot 的 index
    idx = next((i for i, s in enumerate(slots) if s.get("slot_id") == target_slot_id), None)
    if idx is None:
        return None

    target = slots[idx]
    target_start = target.get("start_time")
    target_dur = target.get("duration_minutes") or 90

    prev_slot = slots[idx - 1] if idx > 0 else None
    next_slot = slots[idx + 1] if idx < len(slots) - 1 else None

    from datetime import datetime, timedelta

    def parse_t(t: str | None):
        if not t:
            return None
        try:
            return datetime.strptime(t, "%H:%M")
        except ValueError:
            return None

    target_dt = parse_t(target_start)
    conflicts = []

    # 檢查前一個 slot → target
    if prev_slot and target_dt:
        prev_end_dt = parse_t(prev_slot.get("start_time"))
        if prev_end_dt:
            prev_end_dt += timedelta(minutes=prev_slot.get("duration_minutes") or 90)
            available = (target_dt - prev_end_dt).total_seconds() / 60
            needed = travel_min(prev_slot.get("shop_name", ""), target.get("shop_name", ""))
            if available < needed:
                conflicts.append(f"從 {prev_slot.get('shop_name')} 過來需要 {needed} 分鐘，但只有 {int(available)} 分鐘")

    # 檢查 target → 下一個 slot
    if next_slot and target_dt:
        next_start_dt = parse_t(next_slot.get("start_time"))
        if next_start_dt:
            target_end_dt = target_dt + timedelta(minutes=target_dur)
            available = (next_start_dt - target_end_dt).total_seconds() / 60
            needed = travel_min(target.get("shop_name", ""), next_slot.get("shop_name", ""))
            if available < needed:
                conflicts.append(f"到 {next_slot.get('shop_name')} 需要 {needed} 分鐘，但只有 {int(available)} 分鐘")

    if not conflicts:
        return None

    # 計算可行時間窗口
    feasible_window = None
    if prev_slot and next_slot:
        prev_end_dt = parse_t(prev_slot.get("start_time"))
        next_start_dt = parse_t(next_slot.get("start_time"))
        if prev_end_dt and next_start_dt:
            prev_end_dt += timedelta(minutes=prev_slot.get("duration_minutes") or 90)
            travel_from_prev = travel_min(prev_slot.get("shop_name", ""), target.get("shop_name", ""))
            travel_to_next = travel_min(target.get("shop_name", ""), next_slot.get("shop_name", ""))
            earliest = prev_end_dt + timedelta(minutes=travel_from_prev)
            latest = next_start_dt - timedelta(minutes=travel_to_next + target_dur)
            if earliest <= latest:
                feasible_window = {
                    "earliest": earliest.strftime("%H:%M"),
                    "latest": latest.strftime("%H:%M"),
                }

    return {
        "conflict_type": "transport",
        "slot_id": target_slot_id,
        "prev_shop": prev_slot.get("shop_name") if prev_slot else None,
        "next_shop": next_slot.get("shop_name") if next_slot else None,
        "feasible_window": feasible_window,
        "message": "；".join(conflicts),
    }


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

    # 1. 追問選項攔截器
    if state.get("awaiting_intent_clarification"):
        q_upper = q.upper()
        city_match = None
        if "A" in q_upper or "東京" in q: city_match = "東京"
        elif "B" in q_upper or "大阪" in q: city_match = "大阪"
        elif "C" in q_upper or "京都" in q: city_match = "京都"
        elif "D" in q_upper or "台北" in q: city_match = "台北"

        if city_match:
            curr_intent = state.get("intent") or {}
            if not isinstance(curr_intent, dict):
                curr_intent = curr_intent.as_dict() if hasattr(curr_intent, "as_dict") else {}
            curr_intent["city"] = city_match
            curr_intent["is_actionable"] = True
            
            state["intent"] = curr_intent
            state["awaiting_intent_clarification"] = False
            state["clarification_broadcast"] = None
            return state


    if _consume_pending_dietary_clarification_answer(state, _svc_llm_router(state)):
        return state
    if _consume_pending_intent_clarification(state, _svc_llm_router(state)):
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
            global_schedule=state.get("global_schedule"),
            itinerary_slots=list(state.get("itinerary_slots") or []),
        )
        # ▼▼▼ [新增 DEBUG 1：印出 LLM 解析出的完整 JSON] ▼▼▼
        print("\n" + "="*50)
        print("👉 [DEBUG] LLM 解析出的 Intent 原始內容：")
        print(json.dumps(intent.as_dict(), indent=2, ensure_ascii=False))
        print("="*50 + "\n")
        # ▲▲▲ ===================================== ▲▲▲
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

    # Execute slot locking from revision_op (set by LLM or rule path)
    rev_op = state["intent"].get("revision_op")
    if rev_op and isinstance(rev_op, dict):
        target_shop = rev_op.get("target_shop") or ""
        slots = list(state.get("itinerary_slots") or [])
        if slots and target_shop:
            # Back-fill slot_id into intent
            for slot in slots:
                if slot.get("shop_name") == target_shop:
                    state["intent"]["revision_op"]["slot_id"] = slot["slot_id"]
                    break
            # Clear any previous session_locked before applying new lock
            for slot in slots:
                slot["session_locked"] = False
            # Lock all slots except the target
            updated_slots = []
            for slot in slots:
                s = dict(slot)
                s["session_locked"] = (s.get("shop_name") != target_shop)
                # user_locked stays as is
                updated_slots.append(s)
            state["itinerary_slots"] = updated_slots

            # Conflict detection
            slot_id = rev_op.get("slot_id")
            if slot_id:
                catalog = _get_shop_catalog(state)
                conflict = _conflict_check_travel(
                    updated_slots,
                    slot_id,
                    catalog,
                )
                if conflict:
                    state["conflict"] = conflict
                    fw = conflict.get("feasible_window")
                    if fw:
                        msg = (
                            f"交通時間有點趕：{conflict['message']}。\n"
                            f"{fw['earliest']}～{fw['latest']} 到都來得及，"
                            f"你想要什麼時候去？"
                        )
                    else:
                        msg = (
                            f"交通時間有點趕：{conflict['message']}。\n"
                            f"建議調整行程順序或時間。"
                        )
                    state["clarification_broadcast"] = {
                        "type": "travel_conflict",
                        "question": msg,
                        "hint": "travel_time",
                        "feasible_window": conflict.get("feasible_window"),
                    }

    # Apply user_locked from must_include_shops
    must_include = state["intent"].get("must_include_shops") or []
    if must_include:
        slots = list(state.get("itinerary_slots") or [])
        for slot in slots:
            if slot.get("shop_name") in must_include:
                slot["user_locked"] = True
        state["itinerary_slots"] = slots

    # Apply user_locked from must_exclude_shops (unlock)
    must_exclude = state["intent"].get("must_exclude_shops") or []
    if must_exclude:
        slots = list(state.get("itinerary_slots") or [])
        for slot in slots:
            if slot.get("shop_name") in must_exclude:
                slot["user_locked"] = False
        state["itinerary_slots"] = slots

    # excluded_shops 觸發排除時，沒被點名的 slot 視為鎖定（partial slot invariance）
    excluded_shop_names_for_lock = set(state["intent"].get("excluded_shops") or [])
    rev_op_for_lock = state["intent"].get("revision_op")
    if state["intent"].get("is_revision") and excluded_shop_names_for_lock and not rev_op_for_lock:
        slots = list(state.get("itinerary_slots") or [])
        if slots:
            target_slot_ids = {
                s.get("slot_id") for s in slots
                if s.get("shop_name") in excluded_shop_names_for_lock
            }
            if target_slot_ids:
                updated_slots = []
                for slot in slots:
                    s = dict(slot)
                    if s.get("slot_id") not in target_slot_ids:
                        s["session_locked"] = True
                    updated_slots.append(s)
                state["itinerary_slots"] = updated_slots

    if not intent.is_actionable:
        msg = (intent.actionability_followup or "").strip()

        # ▼▼▼ [新增 DEBUG 2：看看準備廣播的 msg 長怎樣] ▼▼▼
        print(f"👉 [DEBUG] 準備廣播的追問訊息: {msg!r}")
        # ▲▲▲ ===================================== ▲▲▲
        
        if not msg:
            msg = "請告訴我您具體想去的城市或國家，以便我為您規劃。"
        state["awaiting_intent_clarification"] = True
        state["clarification_broadcast"] = {"type": "clarification", "question": msg}
        state["plan_excluded_shop_tags"] = []
        state["wants_flight_search"] = False
        state.setdefault("research_log", []).append(
            _dj("intent_not_actionable", excerpt=msg[:200])
        )
        return state
    state["awaiting_intent_clarification"] = False
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

    intent_dict = state.get("intent") or {}
    meal_slots = intent_dict.get("meal_slots") or []
    cache = dict(state.get("candidate_cache") or {})
    all_hit = all(
        _make_cache_key(mt, intent_dict) in cache
        for mt in meal_slots
    )
    if all_hit and meal_slots:
        candidate_names = []
        for mt in meal_slots:
            key = _make_cache_key(mt, intent_dict)
            candidates = cache.get(key) or []
            if candidates:
                candidate_names.append(candidates[0])
        state["researcher_candidate_names"] = candidate_names
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

    # Write to candidate cache
    candidate_names = state.get("researcher_candidate_names") or []
    cache = dict(state.get("candidate_cache") or {})
    for mt, name in zip(meal_slots, candidate_names):
        key = _make_cache_key(mt, intent_dict)
        if key not in cache:
            cache[key] = [name]
        elif name not in cache[key]:
            cache[key].append(name)
    state["candidate_cache"] = cache

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
                temperature=0.0,
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




@trace_agent_stage("planner")
async def node_plan(state: AgentState, config: Optional[RunnableConfig] = None) -> AgentState:
    print(_dj("debug_print", node="node_plan", message="Generating outcome report"))
    if state.get("error"):
        out = _finalize_plan_on_agent_error(state)
        query = state.get("query") or ""
        entry_type = "user_turn"
        extend_turn_checkpoint_in_state(out, config, entry_type=entry_type, description=query[:80])
        return out
    try:
        out = await _node_plan_core(state)
    except Exception as exc:
        logger.exception("node_plan failed agent_run_id=%s", state.get("agent_run_id"))
        _attach_node_error(state, "plan", exc)
        out = _finalize_plan_on_agent_error(state)

    query = state.get("query") or ""

    # Router：判斷 checkpoint entry type
    intent_dict = out.get("intent") or {}
    is_revision = intent_dict.get("is_revision", False)
    confirm_op = intent_dict.get("confirm_op")

    if confirm_op:
        entry_type = "user_confirmed"
        description = confirm_op.get("message") or query[:80]
    elif is_revision and (out.get("itinerary_slots") != state.get("itinerary_slots")):
        entry_type = "slot_change"
        changed = [
            s["shop_name"] for s in (out.get("itinerary_slots") or [])
            if s not in (state.get("itinerary_slots") or [])
        ]
        description = "換了：" + "、".join(changed[:3]) if changed else query[:80]
    else:
        entry_type = "user_turn"
        description = query[:80]

    # parent_id = 上一個 checkpoint 的 id
    prev_entries = entries_from_state(out)
    parent_id = prev_entries[-1].get_id() if prev_entries else None

    extend_turn_checkpoint_in_state(
        out, config,
        entry_type=entry_type,
        description=description,
        parent_id=parent_id,
    )

    # 更新 shop_checkpoint_index
    new_entries = entries_from_state(out)
    if new_entries:
        new_cp_id = new_entries[-1].get_id()
        idx = dict(out.get("shop_checkpoint_index") or {})
        for slot in (out.get("itinerary_slots") or []):
            shop = slot.get("shop_name") or ""
            if shop:
                if shop not in idx:
                    idx[shop] = []
                if not idx[shop] or idx[shop][-1] != new_cp_id:
                    idx[shop].append(new_cp_id)
        out["shop_checkpoint_index"] = idx

    return out


async def _node_plan_core(state: AgentState) -> AgentState:
    # ------------------------------------------------------------------
    # DP planner – uses actual DP solver for unlocked slots.
    # Locked slots are kept unchanged. `_schedule_slots` assigns times.
    # ------------------------------------------------------------------
    intent_dict = state.get("intent") or {}
    meal_slots_from_intent = intent_dict.get("meal_slots") or []
    existing_slots = list(state.get("itinerary_slots") or [])
    excluded_shop_names = set(intent_dict.get("excluded_shops") or [])

    # 黑名單優先於鎖定：如果一個已鎖定的 slot，它的店家被加進黑名單，
    # 解除這個 slot 的鎖定，讓它回到「可以被 DP 重排」的狀態
    if excluded_shop_names:
        for slot in existing_slots:
            if slot.get("shop_name") in excluded_shop_names:
                slot["session_locked"] = False
                slot["user_locked"] = False

    locked_slots = [s for s in existing_slots if s.get("session_locked") is True or s.get("user_locked") is True]

    # ---------- 1. candidate pool ----------
    seed_profiles = _researcher_seed_catalog(state)
    region = intent_dict.get("region") or "jp"
    dynamic_profiles = [_build_dynamic_shop_profile(p, region=region)
                        for p in (state.get("dynamic_shop_pool") or [])]
    candidate_pool = seed_profiles + dynamic_profiles

    if excluded_shop_names:
        candidate_pool = [s for s in candidate_pool if s.name not in excluded_shop_names]

    # ---------- A1: ensure trip feature scaling exists (initial turn) ----------
    trip_feat = state.get("phase_a_trip_feature_scaling")
    if trip_feat is None:
        ctx_for_scale = {"preferred_tags": list(intent_dict.get("category_tags") or [])}
        fm, fs = compute_trip_frozen_scaling(candidate_pool, ctx_for_scale)
        trip_feat = {"means": [float(x) for x in fm], "stds": [float(x) for x in fs]}
        state["phase_a_trip_feature_scaling"] = trip_feat
        state.setdefault("transit_audit", []).append(
            _dj(
                "A1_trip_feature_scaling_initialized",
                feature_means=trip_feat["means"],
                feature_stds=trip_feat["stds"],
                candidate_count=len(candidate_pool),
            )
        )

    # ---------- 2. Rank candidates ----------
    preference = UserPreference(
        preferred_tags=list(intent_dict.get("category_tags") or []),
        avoid_tags=[],
        dietary_preference=(state.get("dietary_profile") or {}).get("ethics") or "regular",
        max_wait_minutes=35,
        health_budget_limit=1.2,
        max_total_minutes=120,
        max_budget_impact=0.75,
    )
    minefield = UserMinefield()

    mu_vec = state.get("phase_a_posterior_mu")
    if mu_vec is None:
        mu_vec = [0.0] * 6
    else:
        mu_vec = [float(x) for x in mu_vec]
    ctx_features = {"preferred_tags": list(intent_dict.get("category_tags") or [])}

    if intent_dict.get("is_revision"):
        # ---- B1: full C0(s) scoring ---- #
        full_ranked, _ = RankingEngine.generate_top_picks(
            candidate_pool, preference, minefield, return_full=True
        )
        # ---- B1: freeze S0 scaling and feature scaling for this turn ---- #
        turn_id = state.get("agent_run_id") or uuid.uuid4().hex
        rev_op = intent_dict.get("revision_op") or {}
        slot_id = rev_op.get("slot_id")
        trip_feat = state.get("phase_a_trip_feature_scaling") or {"means": [0.0]*6, "stds": [1.0]*6}
        phase_b_context = freeze_phase_b_turn_context(
            full_ranked, ctx_features, slot_id=slot_id, turn_id=turn_id,
            feature_means=trip_feat["means"], feature_stds=trip_feat["stds"],
        )
        state["phase_b_turn_context"] = phase_b_context

        # ---- B2: posterior rerank on full C0(s), then top-M ---- #
        M = 5
        reranked_full = phase_b_rerank(full_ranked, mu_vec, phase_b_context, ctx_features)
        ranked = reranked_full[:M]

        # ---- B3: contender set on top-M ---- #
        _sigma_default = np.eye(6).tolist()
        Sigma_mat = state.get("phase_a_posterior_sigma", _sigma_default)
        if Sigma_mat is None:
            Sigma_mat = _sigma_default
        _contender, _meta = contender_set(
            ranked, mu_vec, Sigma_mat, ctx_features, phase_b_context
        )
        state["phase_b_contender_size"] = int(_meta["size"])
        state["phase_b_contender_meta"] = _meta
        state["phase_b_gate_skipped"] = bool(_meta["gate_short_circuit"])
        state["phase_b_mc_calls"] = _meta["mc_calls"]

        # ---- C1: generate cross-block question candidates (only when not singleton) ---- #
        _xe = state.get("phase_c_event_xe")
        if not state["phase_b_gate_skipped"] and _xe is not None:
            _c1_questions, _ask_eligible = generate_cross_block_questions(
                Sigma=Sigma_mat,
                L_j=_meta.get("L_j") or [0.0] * 6,
                x_e=_xe,
            )
            if not _ask_eligible:
                _c1_questions = []
            state["phase_b_contender_meta"]["c1_questions"] = _c1_questions
            state["phase_b_contender_meta"]["c1_ask_eligible"] = bool(_ask_eligible)

            # ---- C2: compute EVOI for each candidate question (max 3) ---- #
            _c1_evoi_results: list[dict[str, object]] = []
            if _c1_questions:
                _c1_evoi_results = compute_evoi_for_questions(
                    questions=_c1_questions,
                    ranked=ranked,
                    mu=mu_vec,
                    Sigma=Sigma_mat,
                    phase_b_context=phase_b_context,
                    x_e=_xe,
                    evidences=[
                        EvidenceRecord(**row) for row in (state.get("phase_a_evidence_log") or [])
                    ],
                    c_int=C_INT,
                    mc_draws=200,
                    ctx=ctx_features,
                )
                state["phase_b_contender_meta"]["c1_evoi"] = _c1_evoi_results

                # ---- EVOI magnitude distribution diagnostic (gross + net) ---- #
                evoi_values = state.setdefault("phase_c_evoi_values", [])
                for rec in _c1_evoi_results:
                    evoi_values.append(float(rec.get("gross_evoi", rec.get("evoi", 0.0))))
                if len(evoi_values) >= 20:
                    import statistics
                    gross_vals = [float(x.get("gross_evoi", x.get("evoi", 0.0))) for x in _c1_evoi_results]
                    net_vals = [float(x.get("net_evoi", x.get("evoi", 0.0))) for x in _c1_evoi_results]
                    print(
                        "EVOI_DIAGNOSTICS "
                        f"gross_min={min(gross_vals):.6f} gross_max={max(gross_vals):.6f} "
                        f"gross_median={statistics.median(gross_vals):.6f} "
                        f"net_min={min(net_vals):.6f} net_max={max(net_vals):.6f} "
                        f"net_median={statistics.median(net_vals):.6f} "
                        f"c_int={C_INT} count={len(evoi_values)}"
                    )
                    state["phase_c_evoi_values"] = []
            else:
                state["phase_b_contender_meta"]["c1_evoi"] = []

            # ---- C3: gate decision ---- #
            _gate = evaluate_gate(
                ask_eligible=bool(_ask_eligible),
                evoi_results=_c1_evoi_results,
                asked_this_turn=bool(state.get("asked_this_turn", False)),
                contender_size=int(_meta["size"]),
                attribution_already_given=bool(state.get("phase_c_attribution_already_given", False)),
            )
            state["phase_c_gate"] = _gate
        else:
            state["phase_b_contender_meta"]["c1_questions"] = []
            state["phase_b_contender_meta"]["c1_ask_eligible"] = False
            state["phase_b_contender_meta"]["c1_evoi"] = []
            if state.get("phase_b_gate_skipped"):
                state["phase_c_gate"] = {"action": "continue", "reason": "decision_stable"}
            elif state.get("phase_c_attribution_already_given"):
                state["phase_c_gate"] = {"action": "continue", "reason": "attribution_already_given"}
            else:
                state["phase_c_gate"] = {"action": "continue", "reason": "not_block_ambiguous"}

        state.setdefault("transit_audit", []).append(
            _dj(
                "B3_contender_set",
                contender_size=int(_meta["size"]),
                candidate_size=len(ranked),
                mc_calls=state["phase_b_mc_calls"],
                gate_skipped=state["phase_b_gate_skipped"],
                L_j=_meta.get("L_j"),
            )
        )
    else:
        ranked, rejected = RankingEngine.rank(candidate_pool, preference, minefield)

    if not ranked:
        state.setdefault("transit_audit", []).append(
            _dj("dp_planner_no_ranked_shops",
                candidate_count=len(candidate_pool),
                rejected_count=len(rejected))
        )
        state["itinerary_slots"] = existing_slots
        state["final_itinerary"] = "## 行程無法生成\n\n無法取得任何評分候選店家。"
        state["ui_cards"] = []
        return state

    # ---------- 3. DP selection for unlocked slots ----------
    required_length = len(meal_slots_from_intent) - len(locked_slots)

    if required_length <= 0:
        # all locked – keep existing slots unchanged
        itinerary_slots = list(existing_slots)
        state.setdefault("transit_audit", []).append(
            _dj("dp_planner_all_locked", reason="required_length_zero_or_negative")
        )
    else:
        start_time = datetime.now(_APP_TZ)
        traffic = MockTrafficProvider()
        locked_shop_names = {s.get("shop_name") for s in locked_slots if s.get("shop_name")}

        # temporarily patch GraphBuilder.build_graph to use the pinned version
        original_build = GraphBuilder.build_graph
        GraphBuilder.build_graph = _dp_graph_builder_with_itinerary_day_pin
        try:
            graph = GraphBuilder.build_graph(
                ranked, traffic, start_time,
                meal_slots=meal_slots_from_intent,  # full order
                mode=OptimizationMode.BALANCED,
                requested_meal_count=required_length,
                slot_required_tags=None,
                excluded_shop_tags=frozenset(state.get("plan_excluded_shop_tags") or []),
            )
        finally:
            GraphBuilder.build_graph = original_build

        banned_node_ids: set[str] = set()
        for n in graph.nodes:
            if n.shop_name in locked_shop_names:
                banned_node_ids.add(n.node_id)
            if n.shop_name in excluded_shop_names:
                banned_node_ids.add(n.node_id)

        resolved_path = agent_dp_find_optimal_path_no_shop_repeat(
            graph,
            required_length=required_length,
            must_have_tags=set(),
            banned_node_ids=banned_node_ids,
            solver_audit_log=state.setdefault("transit_audit", []),
            excluded_shop_tags=frozenset(state.get("plan_excluded_shop_tags") or []),
        )

        if len(resolved_path) < required_length:
            state.setdefault("transit_audit", []).append(
                _dj("dp_planner_degraded",
                    requested=required_length,
                    achieved=len(resolved_path))
            )

        # ---------- 4. Assemble itinerary_slots ----------
        if locked_slots:
            path_iter = iter(resolved_path)
            itinerary_slots = []
            for slot in existing_slots:
                if slot.get("session_locked") is True or slot.get("user_locked") is True:
                    itinerary_slots.append(dict(slot))
                else:
                    node = next(path_iter, None)
                    if node is None:
                        itinerary_slots.append(dict(slot))
                        continue
                    itinerary_slots.append({
                        "slot_id": str(uuid.uuid4()),
                        "meal_type": slot.get("meal_type", ""),
                        "shop_name": node.shop_name,
                        "user_locked": False,
                        "session_locked": False,
                        "start_time": None,
                        "duration_minutes": 90,
                    })
        else:
            itinerary_slots = []
            for meal_type, node in zip(meal_slots_from_intent, resolved_path):
                itinerary_slots.append({
                    "slot_id": str(uuid.uuid4()),
                    "meal_type": meal_type,
                    "shop_name": node.shop_name,
                    "user_locked": False,
                    "session_locked": False,
                    "start_time": None,
                    "duration_minutes": 90,
                })
            if len(resolved_path) < len(meal_slots_from_intent):
                missing = meal_slots_from_intent[len(resolved_path):]
                state.setdefault("transit_audit", []).append(
                    _dj("dp_planner_missing_slots", missing_meal_types=missing)
                )

    # ---------- 5. Schedule times ----------
    shop_catalog = {}
    for sp in seed_profiles + dynamic_profiles:
        shop_catalog[sp.name] = sp
    itinerary_slots = _schedule_slots(itinerary_slots, shop_catalog)
    state["itinerary_slots"] = itinerary_slots

    # ---------- 6. Build report and ui_cards ----------
    report_lines = [
        "## Travel Agent - Live Run\n",
        "### Itinerary (DP planner)\n",
        "| Time | Shop | Note |\n|---|---|---|\n",
    ]
    locked_names = {s.get("shop_name") for s in locked_slots if s.get("shop_name")}
    for slot in itinerary_slots:
        start_time_field = slot.get("start_time") or "??"
        shop_name = slot.get("shop_name") or "??"
        tag = "LOCKED" if (slot.get("session_locked") or slot.get("user_locked")) else "DP"
        report_lines.append(f"| {start_time_field} | {shop_name} | {tag} |\n")
    report_lines.append("\n### Summary\n")
    report = "".join(report_lines)
    state["final_itinerary"] = report

    MEAL_LABEL = {
        "breakfast": "早餐", "lunch": "午餐", "tea": "下午茶",
        "dinner": "晚餐", "late_night": "宵夜",
    }
    ui_cards = []
    for slot in itinerary_slots:
        meal_label = MEAL_LABEL.get(slot.get("meal_type", ""), slot.get("meal_type", ""))
        ui_cards.append({
            "shop_name": slot.get("shop_name", ""),
            "meal_type": slot.get("meal_type", ""),
            "meal_label": meal_label,
            "slot_id": slot.get("slot_id", ""),
            "user_locked": slot.get("user_locked", False),
            "session_locked": slot.get("session_locked", False),
            "start_time": slot.get("start_time"),
            "duration_minutes": slot.get("duration_minutes", 90),
            "address_hint": "",
            "why_selected": "",
            "how_to_go": "",
            "reservation_hint": "",
            "rank_note": "",
            "insider_pick": False,
            "warning_badge": "",
            "warning_text": "",
            "lat": None,
            "lng": None,
        })
    state["ui_cards"] = ui_cards

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
    route_intent → [if intent not actionable → END with clarification] else clarify_constraint → retriever → …
    clarify_constraint → retriever → researcher → critic ⟲ (request_more → researcher, max-iter guard)
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

    def _after_route_intent(state: AgentState) -> str:
        if state.get("awaiting_intent_clarification"):
            return "end"
        return "clarify_constraint"

    def _after_clarify_constraint(state: AgentState) -> str:
        if state.get("awaiting_dietary_clarification"):
            return "end"
        return "retriever"

    g.add_conditional_edges(
        "route_intent",
        _after_route_intent,
        {"end": END, "clarify_constraint": "clarify_constraint"},
    )
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
        elif (rejected or verdict == "request_more"):
            # Check the critic_retry_count for escalation
            if state.get("critic_retry_count", 0) < 2:
                return "researcher"
            else:
                # Escalation: max retries reached
                state.setdefault("transit_audit", []).append(
                    _dj("critic_escalation", message="Max retries reached, proceeding to plan")
                )
                return "plan"
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
