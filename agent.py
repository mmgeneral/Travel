from __future__ import annotations

import copy
import json
from langgraph.graph import END, StateGraph
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import os
import re
from pathlib import Path
import sqlite3
import uuid

import requests
from debug_json import audit_json_line_as_text, debug_json as _dj
from zoneinfo import ZoneInfo
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
try:
    from langgraph.checkpoint.sqlite import SqliteSaver
except Exception:  # pragma: no cover
    SqliteSaver = None

from duffel import DuffelClient
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
from decision_engine import (
    DensityScanner,
    GraphBuilder,
    HealthBudgetTracker,
    ScoringEngine,
    ItinerarySynthesizer,
    OptimizationMode,
    RankingEngine,
    RankedShop,
    SelectionStrategy,
    UserPreferenceLearner,
    UserMinefield,
    UserPreference,
    WeightProfile,
    choose_health_backup,
)
from intent_parser import parse_intent as _parse_intent, Intent as _Intent
from llm_router import LLMRouter as _LLMRouter


class AgentState(TypedDict):
    query: str
    research_log: list[str]
    transit_audit: list[str]
    final_itinerary: str
    rollback_occurred: bool
    saga_snapshot_idx: int
    leg1_offer: dict
    leg2_offer: dict
    feedback_updates: list[str]
    learned_weight_profile: dict
    agent_run_id: str
    dietary_profile: dict
    conv_saga_path: str
    ui_cards: list[dict]
    advanced_mode: bool
    dynamic_shop_pool: list[dict]
    user_locale: str
    wants_flight_search: bool
    user_lat: float | None
    user_lng: float | None
    researcher_candidate_names: list[str]
    researcher_notes: str
    auditor_feedback: str
    auditor_rejected: bool
    research_iteration: int
    intent: dict  # serialised Intent.as_dict(); populated once in node_route_intent


class AgentStateModel(BaseModel):
    query: str
    research_log: list[str] = Field(default_factory=list)
    transit_audit: list[str] = Field(default_factory=list)
    final_itinerary: str = ""
    rollback_occurred: bool = False
    saga_snapshot_idx: int = -1
    leg1_offer: dict = Field(default_factory=dict)
    leg2_offer: dict = Field(default_factory=dict)
    feedback_updates: list[str] = Field(default_factory=list)
    learned_weight_profile: dict = Field(default_factory=dict)
    agent_run_id: str = ""
    dietary_profile: dict = Field(default_factory=lambda: {"ethics": "unspecified", "allergens": [], "religious": "none", "medical": []})
    conv_saga_path: str = ""
    ui_cards: list[dict] = Field(default_factory=list)
    advanced_mode: bool = False
    dynamic_shop_pool: list[dict] = Field(default_factory=list)
    user_locale: str = ""
    wants_flight_search: bool = False
    user_lat: float | None = None
    user_lng: float | None = None
    researcher_candidate_names: list[str] = Field(default_factory=list)
    researcher_notes: str = ""
    auditor_feedback: str = ""
    auditor_rejected: bool = False
    research_iteration: int = 0
    intent: dict = Field(default_factory=dict)


class AtomicCommitFailure(Exception):
    pass


class OfflineDuffelClient:
    """Minimal offline stub for local demo/self-use mode."""

    @staticmethod
    def search_offers(origin: str, destination: str, date: str) -> dict:
        _ = date
        return {
            "offers": [
                {
                    "id": f"offline_{origin.lower()}_{destination.lower()}",
                    "total_amount": "199.00",
                    "total_currency": "USD",
                    "slices": [
                        {
                            "segments": [
                                {
                                    "origin": {"iata_code": origin},
                                    "destination": {"iata_code": destination},
                                    "operating_carrier": {"iata_code": "OF"},
                                    "operating_carrier_flight_number": "101",
                                    "departing_at": "2026-04-24T09:00:00Z",
                                    "arriving_at": "2026-04-24T12:00:00Z",
                                }
                            ]
                        }
                    ],
                }
            ],
            "passenger_id": "offline_passenger_001",
        }


_SAGA_DIR = Path(os.getenv("SAGA_PERSIST_DIR", str(Path.home() / ".travel_agent" / "saga")))
_SAGA_DIR.mkdir(parents=True, exist_ok=True)
_APP_TZ = ZoneInfo(os.getenv("APP_TIMEZONE", "Asia/Taipei"))
_LEARNING_DB = _SAGA_DIR / "learning_state.db"
_duffel = OfflineDuffelClient() if os.getenv("OFFLINE_MODE", "0") == "1" else DuffelClient()
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
) -> dict:
    run_id = agent_run_id or uuid.uuid4().hex
    model = AgentStateModel(
        query=query,
        dietary_profile=dietary_profile or {"ethics": "unspecified", "allergens": [], "religious": "none", "medical": []},
        agent_run_id=run_id,
        conv_saga_path="",
        advanced_mode=advanced_mode,
        user_locale=(user_locale or "").strip(),
        wants_flight_search=False,
        user_lat=user_lat,
        user_lng=user_lng,
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
    q = (query or "").lower()
    tags: set[str] = set()
    if any(k in q for k in ("拉麵", "拉面", "ramen")):
        tags.add("ramen")
    if any(k in q for k in ("居酒屋", "izakaya")):
        tags.add("izakaya")
    kws_l = [k.lower() for k in _extract_search_category_keywords(query)]
    if any(x in kws_l for x in ("cake", "bakery", "dessert", "patisserie", "coffee", "cafe")):
        tags.add("dessert")
    if any(x in kws_l for x in ("yakitori", "izakaya", "beer", "pub", "japanese pub")):
        tags.add("izakaya")
    if "ramen" in kws_l:
        tags.add("ramen")
    return tags


def _direct_food_category_mentions(query: str) -> set[str]:
    """使用者在本輪 query 中明確說出口的食物類別（不含搜尋擴展／歷史偏好）。"""
    q_raw = query or ""
    q = q_raw.lower()
    out: set[str] = set()
    if any(k in q_raw for k in ("拉麵", "拉面")) or "ramen" in q:
        out.add("ramen")
    if any(
        k in q_raw
        for k in ("蛋糕", "甜點", "甜点", "下午茶", "茶點", "巴斯克", "提拉米蘇", "抹茶", "戚風")
    ):
        out.add("dessert")
    if any(k in q_raw for k in ("串燒", "串烧", "燒鳥", "烧鸟", "焼き鳥")) or "yakitori" in q:
        out.add("izakaya")
    if any(k in q_raw for k in ("居酒屋", "啤酒", "生啤")) or "izakaya" in q:
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


def _seed_covers_meal_slot(seed_shops: list[ShopProfile], slot: str) -> bool:
    """True if at least one seed shop has tags/occasion overlap with slot needs."""
    needed = _SLOT_TAG_COVERAGE.get(slot)
    if not needed:
        return True
    for shop in seed_shops:
        bag = {str(t).lower() for t in shop.tags} | {str(t).lower() for t in shop.occasion_tags}
        if bag & needed:
            return True
    return False


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


def _llm_broad_geo_search_queries(user_query: str, city: str, region: str) -> tuple[list[str], str]:
    """
    Ask an LLM for broader English Places search phrases; fall back to heuristics.
    Returns (queries, mode_tag for audit).
    """
    fb = _fallback_broad_geo_queries(user_query, city, region)
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return fb, "heuristic_fallback_no_api_key"

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
    system_msg = (
        "You fix sparse Google Places Text Search results. Reply with JSON only, no markdown: "
        '{"queries":["..."]} containing 5 to 8 short English search strings. '
        "Broaden geography (neighborhoods, wards, stations, surrounding areas near the city); "
        "relax overly narrow cuisine filters. Do not repeat the same phrase twice."
    )
    user_msg = f"City: {city} (region={region}). User query:\n{(user_query or '')[:2000]}"
    try:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                "temperature": 0.35,
                "max_tokens": 450,
            },
            timeout=45,
        )
        resp.raise_for_status()
        payload = resp.json()
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


def _build_shop_catalog() -> list[ShopProfile]:
    return [
        ShopProfile(
            name="燃えよ麺助",
            close_time="20:30",
            booking_type=BookingType.NONE,
            queue_strategy=QueueStrategy.PHYSICAL_LINE,
            last_call_offset=30,
            is_cash_only=True,
            sns_handle="moeyo_mensuke",
            avg_eat_minutes=55,
            is_famous=True,
            base_wait_minutes=35,
            base_health_impact=0.72,
            customization_score=0.35,
            flavor_intensity=0.92,
            flavor_category=FlavorCategory.HEAVY,
            flavor_vector={"salt": 0.82, "fat": 0.75, "umami": 0.9, "acid": 0.22, "spice": 0.35},
            min_lead_hours=0,
            google_rating=4.35,
            source_scores={"tablelog": 0.82, "sns": 0.76, "google": 0.68},
            authority_data=AuthorityData(
                tablelog_medal="",
                michelin_star=0,
                chef_lineage=[],
                specialty_items=["濃厚豚骨ラーメン"],
                google_reviews=[
                    "スープと麺の一体感が強く、豚骨の旨味がはっきりしている。",
                    "行列はあるが回転も悪くなく、ワンオペでも丁寧。",
                ],
                review_count=180,
            ),
            occasion_tags={"ramen", "main_meal", "umami_heavy"},
            neighborhood="Umeda",
            requires_menu_reservation=False,
            nearby_atm_options=["Seven Bank", "JP Post Bank", "AEON Bank"],
            backup_options=["人類みな麺類"],
            latitude=34.7049,
            longitude=135.4950,
        ),
        ShopProfile(
            name="Harbs 大丸京都",
            close_time="20:00",
            booking_type=BookingType.NONE,
            queue_strategy=QueueStrategy.SIGN_UP_SHEET,
            last_call_offset=45,
            is_cash_only=False,
            sns_handle="harbs_official",
            avg_eat_minutes=50,
            is_famous=True,
            base_wait_minutes=20,
            base_health_impact=0.42,
            customization_score=0.8,
            flavor_intensity=0.55,
            flavor_category=FlavorCategory.SWEET,
            flavor_vector={"salt": 0.15, "fat": 0.62, "umami": 0.2, "acid": 0.35, "spice": 0.05},
            min_lead_hours=1,
            source_scores={"tablelog": 0.66, "sns": 0.61, "google": 0.54},
            occasion_tags={"dessert", "cafe", "afternoon_tea"},
            neighborhood="Karasuma",
            requires_menu_reservation=True,
            allowed_dietary_preferences=["regular", "vegetarian"],
            reservation_channels=["TableCheck", "Tabelog予約"],
            backup_options=["マールブランシュ"],
            latitude=35.0034,
            longitude=135.7594,
        ),
        ShopProfile(
            name="一蘭 京都河原町",
            close_time="23:00",
            booking_type=BookingType.NONE,
            queue_strategy=QueueStrategy.TICKET_SYSTEM,
            last_call_offset=40,
            is_cash_only=False,
            has_small_portion=True,
            portion_strictness=0.8,
            sns_handle="ichiran_kyoto",
            avg_eat_minutes=40,
            is_famous=True,
            base_wait_minutes=25,
            base_health_impact=0.6,
            customization_score=0.55,
            flavor_intensity=0.84,
            flavor_category=FlavorCategory.HEAVY,
            flavor_vector={"salt": 0.68, "fat": 0.58, "umami": 0.78, "acid": 0.18, "spice": 0.25},
            min_lead_hours=0,
            trust_score=0.78,
            source_scores={"tablelog": 0.79, "sns": 0.73, "google": 0.71},
            authority_data=AuthorityData(
                tablelog_medal="",
                michelin_star=0,
                chef_lineage=[],
                specialty_items=["天然豚骨拉麵"],
                google_reviews=[
                    "口味穩定，湯頭濃厚但層次變化有限。",
                    "深夜很方便，但驚喜感不高。",
                    "是安全牌，不是頂級名店線。",
                ],
                review_count=540,
            ),
            occasion_tags={"ramen", "quick_meal", "late_night"},
            neighborhood="Kawaramachi",
            tags=["ramen", "ticket_system", "late_open", "大型連鎖"],
            backup_options=["麺屋優光"],
            latitude=35.0052,
            longitude=135.7677,
        ),
        ShopProfile(
            name="喫茶ソワレ",
            close_time="19:00",
            booking_type=BookingType.NONE,
            queue_strategy=QueueStrategy.PHYSICAL_LINE,
            last_call_offset=20,
            is_cash_only=True,
            sns_handle="soiree_kyoto",
            avg_eat_minutes=35,
            is_famous=True,
            base_wait_minutes=18,
            base_health_impact=0.38,
            customization_score=0.75,
            flavor_intensity=0.35,
            flavor_category=FlavorCategory.REFRESHING,
            flavor_vector={"salt": 0.12, "fat": 0.2, "umami": 0.25, "acid": 0.52, "spice": 0.02},
            min_lead_hours=0,
            trust_score=0.69,
            source_scores={"tablelog": 0.71, "sns": 0.64, "google": 0.59},
            occasion_tags={"dessert", "cafe", "refresh"},
            neighborhood="Kawaramachi",
            tags=["cafe", "cash_only", "retro"],
            nearby_atm_options=["Seven Bank", "JP Post Bank"],
            backup_options=["長楽館カフェ"],
            latitude=35.0039,
            longitude=135.7709,
        ),
        ShopProfile(
            name="朝拉麵",
            close_time="11:00",
            open_time="06:00",
            booking_type=BookingType.NONE,
            queue_strategy=QueueStrategy.PHYSICAL_LINE,
            last_call_offset=20,
            is_cash_only=False,
            has_small_portion=True,
            portion_strictness=0.8,
            sns_handle="asa_ramen_kyoto",
            avg_eat_minutes=35,
            min_eat_minutes=15,
            is_famous=False,
            base_wait_minutes=12,
            base_health_impact=0.42,
            customization_score=0.58,
            flavor_intensity=0.5,
            flavor_category=FlavorCategory.REFRESHING,
            flavor_vector={"salt": 0.36, "fat": 0.3, "umami": 0.52, "acid": 0.28, "spice": 0.12},
            min_lead_hours=0,
            trust_score=0.65,
            source_scores={"google": 0.66, "sns": 0.62},
            authority_data=AuthorityData(
                tablelog_medal="",
                michelin_star=0,
                chef_lineage=[],
                specialty_items=["朝限定醬油拉麵", "清湯叉燒麵"],
                google_reviews=[
                    "清晨營業很方便，湯頭清爽。",
                    "麵量剛好，適合早餐時段。",
                ],
                review_count=42,
            ),
            occasion_tags={"breakfast", "ramen", "quick_meal"},
            neighborhood="Kawaramachi",
            tags=["ramen", "breakfast", "light"],
            backup_options=["喫茶ソワレ"],
            latitude=35.0048,
            longitude=135.7664,
        ),
        ShopProfile(
            name="松籟庵",
            close_time="20:00",
            open_time="11:00",
            booking_type=BookingType.PHONE,
            queue_strategy=QueueStrategy.PHYSICAL_LINE,
            last_call_offset=30,
            is_cash_only=False,
            has_small_portion=False,
            portion_strictness=1.0,
            sns_handle="shoraian_arashiyama",
            avg_eat_minutes=70,
            is_famous=True,
            base_wait_minutes=28,
            base_health_impact=0.5,
            customization_score=0.7,
            flavor_intensity=0.48,
            flavor_category=FlavorCategory.LIGHT,
            flavor_vector={"salt": 0.22, "fat": 0.28, "umami": 0.5, "acid": 0.3, "spice": 0.08},
            min_lead_hours=3,
            trust_score=0.83,
            source_scores={"tablelog": 0.88, "sns": 0.80, "google": 0.72},
            authority_data=AuthorityData(
                tablelog_medal="百名店 / 銅賞",
                michelin_star=1,
                chef_lineage=["京都老舖和久傳修煉"],
                specialty_items=["汲み上げ湯葉", "季節懷石"],
                google_reviews=[
                    "湯葉口感與出汁層次很細緻，餘韻乾淨。",
                    "懷石節奏穩，味型平衡且不膩。",
                    "香氣、鮮味、口感都很完整。",
                ],
                review_count=128,
            ),
            occasion_tags={"kaiseki", "course", "scenic"},
            neighborhood="Arashiyama",
            tags=["kaiseki", "reservation", "scenic"],
            requires_menu_reservation=True,
            allowed_dietary_preferences=["regular", "vegetarian", "pescatarian"],
            booking_phone="+81-75-861-1234",
            reservation_channels=["電話予約", "Concierge代行"],
            backup_options=["嵯峨豆腐稲"],
            latitude=35.0142,
            longitude=135.6730,
        ),
        ShopProfile(
            name="麵屋豬一",
            close_time="21:30",
            open_time="11:30",
            booking_type=BookingType.NONE,
            queue_strategy=QueueStrategy.PHYSICAL_LINE,
            last_call_offset=30,
            is_cash_only=False,
            sns_handle="inoichi_kyoto",
            avg_eat_minutes=45,
            is_famous=True,
            base_wait_minutes=28,
            base_health_impact=0.56,
            customization_score=0.62,
            flavor_intensity=0.46,
            flavor_category=FlavorCategory.LIGHT,
            flavor_vector={"salt": 0.34, "fat": 0.26, "umami": 0.58, "acid": 0.22, "spice": 0.12},
            min_lead_hours=0,
            trust_score=0.84,
            source_scores={"tablelog": 0.9, "sns": 0.79, "google": 0.75},
            authority_data=AuthorityData(
                tablelog_medal="百名店",
                michelin_star=0,
                chef_lineage=["京都淡麗拉麵系統修業"],
                specialty_items=["白湯和出汁拉麵", "炙燒叉燒飯"],
                google_reviews=[
                    "湯頭乾淨有層次，入口鮮味明確。",
                    "麵條口感偏細直，搭配清爽不膩。",
                    "整體屬於淡麗系，回訪率很高。",
                ],
                review_count=266,
            ),
            occasion_tags={"ramen", "lunch", "dinner", "main_meal", "quick_meal"},
            neighborhood="Kawaramachi",
            tags=["ramen", "light", "queue"],
            backup_options=["麺屋優光"],
            latitude=35.0031,
            longitude=135.7649,
        ),
        ShopProfile(
            name="三嶋亭",
            close_time="21:00",
            booking_type=BookingType.PHONE,
            queue_strategy=QueueStrategy.SIGN_UP_SHEET,
            last_call_offset=45,
            is_cash_only=False,
            sns_handle="mishimatei_kyoto",
            avg_eat_minutes=90,
            min_eat_minutes=45,
            is_famous=True,
            base_wait_minutes=25,
            base_health_impact=0.82,
            customization_score=0.38,
            flavor_intensity=0.93,
            flavor_category=FlavorCategory.HEAVY,
            flavor_vector={"salt": 0.48, "fat": 0.9, "umami": 0.88, "acid": 0.1, "spice": 0.08},
            min_lead_hours=2,
            trust_score=0.86,
            source_scores={"tablelog": 0.87, "sns": 0.73, "google": 0.71},
            authority_data=AuthorityData(
                tablelog_medal="老舗名店",
                michelin_star=0,
                chef_lineage=["明治時代壽喜燒老鋪傳承"],
                specialty_items=["關西風壽喜燒", "黑毛和牛割下"],
                google_reviews=[
                    "油脂香氣非常厚實，甜鹹平衡鮮明。",
                    "肉質入口即化，屬重口高滿足感。",
                    "老舖服務節奏穩，值得特地造訪。",
                ],
                review_count=481,
            ),
            occasion_tags={"sukiyaki", "dinner", "course", "social", "main_meal"},
            neighborhood="Kawaramachi",
            tags=["sukiyaki", "heavy", "classic"],
            requires_menu_reservation=True,
            booking_phone="+81-75-221-0003",
            reservation_channels=["電話予約", "Concierge代行"],
            backup_options=["モリタ屋 木屋町店"],
            latitude=35.0058,
            longitude=135.7687,
        ),
        ShopProfile(
            name="和久傳",
            close_time="20:30",
            booking_type=BookingType.PHONE,
            queue_strategy=QueueStrategy.PHYSICAL_LINE,
            last_call_offset=40,
            is_cash_only=False,
            has_small_portion=False,
            portion_strictness=1.0,
            sns_handle="wakuden_kyoto",
            avg_eat_minutes=100,
            is_famous=True,
            base_wait_minutes=18,
            base_health_impact=0.44,
            customization_score=0.69,
            flavor_intensity=0.4,
            flavor_category=FlavorCategory.LIGHT,
            flavor_vector={"salt": 0.24, "fat": 0.22, "umami": 0.66, "acid": 0.28, "spice": 0.05},
            min_lead_hours=3,
            trust_score=0.92,
            source_scores={"tablelog": 0.9, "sns": 0.84, "google": 0.78},
            authority_data=AuthorityData(
                tablelog_medal="百名店 / 銀賞",
                michelin_star=2,
                chef_lineage=["京都料亭和久傳本店修業"],
                specialty_items=["季節懷石", "炭火椀物"],
                google_reviews=[
                    "風味非常細緻，出汁與食材平衡優雅。",
                    "口味偏清麗，層次深且收尾乾淨。",
                    "每道菜節奏完整，細節打磨到位。",
                ],
                review_count=312,
            ),
            occasion_tags={"kaiseki", "dinner", "course", "scenic"},
            neighborhood="Karasuma",
            tags=["kaiseki", "light", "fine_dining"],
            requires_menu_reservation=True,
            allowed_dietary_preferences=["regular", "pescatarian", "vegetarian"],
            booking_phone="+81-75-361-0079",
            reservation_channels=["電話予約", "Concierge代行"],
            backup_options=["菊乃井 本店"],
            latitude=35.0015,
            longitude=135.7582,
        ),
        ShopProfile(
            name="Wildcard Izakaya",
            close_time="01:00",
            booking_type=BookingType.WEB,
            queue_strategy=QueueStrategy.TICKET_SYSTEM,
            last_call_offset=45,
            is_cash_only=False,
            is_sharing_friendly=True,
            portion_strictness=0.3,
            sns_handle="wildcard_izakaya",
            avg_eat_minutes=60,
            is_famous=False,
            base_wait_minutes=12,
            base_health_impact=0.68,
            customization_score=0.4,
            flavor_intensity=0.78,
            flavor_category=FlavorCategory.HEAVY,
            flavor_vector={"salt": 0.58, "fat": 0.68, "umami": 0.62, "acid": 0.2, "spice": 0.4},
            min_lead_hours=0,
            trust_score=0.62,
            source_scores={"tablelog": 0.58, "sns": 0.52, "google": 0.49},
            occasion_tags={"izakaya", "nightlife", "social"},
            neighborhood="Kawaramachi",
            tags=["izakaya", "nightlife", "experimental"],
            backup_options=["鳥貴族 河原町店"],
            latitude=35.0047,
            longitude=135.7681,
        ),
    ]


def _build_shop_catalog_taipei() -> list[ShopProfile]:
    return [
        ShopProfile(
            name="林東芳牛肉麵",
            close_time="23:00",
            booking_type=BookingType.NONE,
            queue_strategy=QueueStrategy.PHYSICAL_LINE,
            last_call_offset=20,
            is_cash_only=False,
            sns_handle="lindongfang_beef",
            avg_eat_minutes=40,
            is_famous=True,
            base_wait_minutes=28,
            flavor_intensity=0.82,
            flavor_category=FlavorCategory.HEAVY,
            flavor_vector={"salt": 0.65, "fat": 0.52, "umami": 0.88, "acid": 0.15, "spice": 0.28},
            source_scores={"google": 0.86, "dcard": 0.72, "threads": 0.64},
            occasion_tags={"beef_noodle", "main_meal", "night_food"},
            neighborhood="Zhongshan",
            region="tw",
            closed_weekdays={1},  # Tue
            backup_options=["門前隱味牛肉麵"],
            latitude=25.0517,
            longitude=121.5414,
        ),
        ShopProfile(
            name="阜杭豆漿",
            close_time="12:30",
            booking_type=BookingType.NONE,
            queue_strategy=QueueStrategy.PHYSICAL_LINE,
            last_call_offset=30,
            is_cash_only=False,
            sns_handle="fuhang_official",
            avg_eat_minutes=30,
            is_famous=True,
            base_wait_minutes=40,
            flavor_intensity=0.46,
            flavor_category=FlavorCategory.LIGHT,
            flavor_vector={"salt": 0.35, "fat": 0.32, "umami": 0.41, "acid": 0.1, "spice": 0.05},
            source_scores={"google": 0.82, "dcard": 0.78, "threads": 0.58},
            occasion_tags={"breakfast", "soy_milk", "local_classic"},
            neighborhood="Zhongzheng",
            region="tw",
            closed_weekdays={0},  # Mon
            backup_options=["永和豆漿大王"],
            latitude=25.0452,
            longitude=121.5232,
        ),
        ShopProfile(
            name="雙月食品社 青島店",
            close_time="21:00",
            booking_type=BookingType.NONE,
            queue_strategy=QueueStrategy.SIGN_UP_SHEET,
            last_call_offset=25,
            is_cash_only=False,
            sns_handle="moonmoonfood",
            avg_eat_minutes=45,
            is_famous=True,
            base_wait_minutes=24,
            flavor_intensity=0.52,
            flavor_category=FlavorCategory.REFRESHING,
            flavor_vector={"salt": 0.3, "fat": 0.22, "umami": 0.57, "acid": 0.18, "spice": 0.12},
            source_scores={"google": 0.84, "dcard": 0.75, "threads": 0.62},
            occasion_tags={"soup", "comfort_food", "family"},
            neighborhood="Zhongzheng",
            region="tw",
            closed_weekdays=set(),
            backup_options=["雞湯大叔 信義店"],
            latitude=25.0442,
            longitude=121.5255,
        ),
        ShopProfile(
            name="阿宗麵線 西門店",
            close_time="22:30",
            booking_type=BookingType.NONE,
            queue_strategy=QueueStrategy.TICKET_SYSTEM,
            last_call_offset=15,
            is_cash_only=False,
            sns_handle="azong_noodle",
            avg_eat_minutes=25,
            is_famous=True,
            base_wait_minutes=18,
            flavor_intensity=0.58,
            flavor_category=FlavorCategory.HEAVY,
            flavor_vector={"salt": 0.5, "fat": 0.25, "umami": 0.62, "acid": 0.2, "spice": 0.22},
            source_scores={"google": 0.79, "dcard": 0.66, "threads": 0.57},
            occasion_tags={"snack", "street_food", "quick_meal"},
            neighborhood="Ximending",
            region="tw",
            closed_weekdays=set(),
            backup_options=["陳記腸蚵麵線"],
            latitude=25.0438,
            longitude=121.5071,
        ),
    ]


def _build_shop_catalog() -> list[ShopProfile]:
    """JSON-backed Kyoto seed catalog thin wrapper."""
    return load_shop_catalog("kyoto")


def _build_shop_catalog_taipei() -> list[ShopProfile]:
    """JSON-backed Taipei seed catalog thin wrapper."""
    return load_shop_catalog("taipei")


def _build_shop_catalog_tokyo() -> list[ShopProfile]:
    """JSON-backed Tokyo seed catalog thin wrapper."""
    return load_shop_catalog("tokyo")


def _reliability_cutoff_for_region(region: str) -> float:
    # Taiwan has denser Google coverage; avoid over-penalizing normal variance.
    if region == "tw":
        return 53.0
    return 60.0


def node_route_intent(state: AgentState) -> AgentState:
    """Parse intent once, store in state, and set routing flags."""
    state["saga_snapshot_idx"] = -1
    q = state.get("query", "") or ""
    intent = _parse_intent(
        q,
        _llm_router,
        user_locale=state.get("user_locale") or None,
        user_lat=state.get("user_lat"),
        user_lng=state.get("user_lng"),
    )
    state["intent"] = intent.as_dict()
    state["wants_flight_search"] = intent.wants_flight
    return state


def node_flight_search(state: AgentState) -> AgentState:
    """Duffel flight offers only (no Places / dynamic pool)."""
    print(_dj("debug_print", node="node_flight_search", message="Duffel TPE NRT SFO"))
    date = "2026-04-24"

    def first_offer(origin: str, destination: str) -> tuple[dict, bool]:
        try:
            result = _duffel.search_offers(origin, destination, date)
            cached = False
        except Exception as exc:
            fallback = OfflineDuffelClient.search_offers(origin, destination, date)
            state.setdefault("transit_audit", []).append(
                _dj(
                    "duffel_search_fallback",
                    origin=origin,
                    destination=destination,
                    error_type=exc.__class__.__name__,
                )
            )
            result = fallback
            cached = True
        offers = result.get("offers", [])
        if not offers:
            return {}, False
        return offers[0], cached

    def normalize_offer(raw: dict) -> dict:
        slices = raw.get("slices", [])
        seg = slices[0]["segments"][0] if slices and slices[0].get("segments") else {}
        origin = seg.get("origin", {}).get("iata_code") or "?"
        destination = seg.get("destination", {}).get("iata_code") or "?"
        carrier = (
            seg.get("operating_carrier", {}).get("iata_code")
            or seg.get("marketing_carrier", {}).get("iata_code")
            or ""
        )
        number = (
            seg.get("operating_carrier_flight_number")
            or seg.get("marketing_carrier_flight_number")
            or ""
        )
        amount = raw.get("total_amount", 0)
        try:
            price = float(amount)
        except (TypeError, ValueError):
            price = 0.0
        return {
            "origin": origin,
            "destination": destination,
            "departure": seg.get("departing_at", ""),
            "arrival": seg.get("arriving_at", ""),
            "price": price,
            "currency": raw.get("total_currency", "USD"),
            "seats": 9,
            "carrier": carrier,
            "flight_num": f"{carrier}{number}" if (carrier or number) else "UNKNOWN",
            "offer_id": raw.get("id", ""),
        }

    leg1_raw, hit1 = first_offer("TPE", "NRT")
    leg2_raw, hit2 = first_offer("NRT", "SFO")
    leg1 = normalize_offer(leg1_raw) if leg1_raw else {}
    leg2 = normalize_offer(leg2_raw) if leg2_raw else {}

    def summarise(offer: dict, cached: bool) -> str:
        if not offer:
            return "no results"
        tag = " [cache]" if cached else " [live]"
        return f"{offer['flight_num']} seats={offer['seats']} ${offer['price']} {offer['currency']}{tag}"

    state["research_log"].append(
        _dj("research_flight_leg", route="TPE->NRT", summary=summarise(leg1, hit1))
    )
    state["research_log"].append(
        _dj("research_flight_leg", route="NRT->SFO", summary=summarise(leg2, hit2))
    )
    state["leg1_offer"] = leg1
    state["leg2_offer"] = leg2
    return state


def node_food_search(state: AgentState) -> AgentState:
    """Google Places / dynamic shop pool; no Duffel calls."""
    print(_dj("debug_print", node="node_food_search", message="Places dynamic pool"))
    query_full = state.get("query", "") or ""
    _intent = state.get("intent") or {}
    city = _intent.get("city") or "京都"
    region = _intent.get("region") or "jp"
    if city == "台北":
        seed_for_coverage = list(_build_shop_catalog_taipei())
    elif city == "東京":
        seed_for_coverage = list(_build_shop_catalog_tokyo())
    else:
        seed_for_coverage = list(_build_shop_catalog())
    search_queries, uncovered_slots = _plan_dynamic_place_queries(query_full, city, seed_for_coverage)

    nearby_tool = NearbySearchTool()
    merged_by_name: dict[str, dict] = {}
    must_have_search = sorted(_plan_global_explicit_tags(query_full))
    for qstr in search_queries:
        batch = nearby_tool.search_places(
            city=city,
            user_query=qstr,
            limit=60,
            must_have_tags=must_have_search if must_have_search else None,
        )
        for p in batch:
            nk = (p.get("name") or "").strip().lower()
            if nk and nk not in merged_by_name:
                merged_by_name[nk] = p
        if len(merged_by_name) >= 60:
            break
    places = list(merged_by_name.values())[:60]

    transit = state.setdefault("transit_audit", [])
    if uncovered_slots:
        transit.append(
            _dj(
                "dynamic_search_forced_slot_gaps",
                uncovered_slots=uncovered_slots,
                detail="seed pool missing tags; extra Places queries required",
            )
        )
    transit.append(
        _dj(
            "dynamic_place_query_plan",
            variants=len(search_queries),
            merged_unique=len(places),
            city=city,
            slot_triggered=_intent.get("meal_slots", []),
        )
    )

    dynamic_pool: list[dict] = []
    for p in places:
        row = _dynamic_pool_row_from_place(p, region)
        dynamic_pool.append(row)
        if row.get("time_unknown"):
            state["transit_audit"].append(
                _dj(
                    "time_unknown_open_time",
                    shop=row["name"],
                    detail="missing open_time from Places source",
                )
            )
    state["dynamic_shop_pool"] = dynamic_pool
    state["research_log"].append(
        _dj("dynamic_place_search_complete", city=city, candidates=len(dynamic_pool))
    )
    return state


def _researcher_shop_pool(state: AgentState) -> list[ShopProfile]:
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
    return out


def _call_researcher_prompt(
    query: str, shops: list[ShopProfile], auditor_feedback: str, iteration: int
) -> tuple[list[str], str]:
    """Prompt-driven candidate generation. Falls back to deterministic heuristic."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    sample = [
        {
            "name": s.name,
            "tags": list(s.tags),
            "open_time": s.open_time,
            "close_time": s.close_time,
            "lat": s.latitude,
            "lng": s.longitude,
            "rating": s.google_rating,
        }
        for s in shops[:25]
    ]
    if api_key:
        try:
            sys_msg = (
                "You are Researcher Agent (student). Build a foodie itinerary candidate list. "
                "Return JSON only: {\"candidate_names\":[...],\"notes\":\"...\"}. "
                "Respect auditor feedback if provided."
            )
            user_msg = json.dumps(
                {
                    "query": query,
                    "iteration": iteration,
                    "auditor_feedback": auditor_feedback,
                    "shops": sample,
                },
                ensure_ascii=False,
            )
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    "messages": [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}],
                    "temperature": 0.2,
                    "max_tokens": 450,
                },
                timeout=35,
            )
            resp.raise_for_status()
            txt = str(resp.json()["choices"][0]["message"]["content"] or "").strip()
            txt = re.sub(r"^```(?:json)?\s*", "", txt)
            txt = re.sub(r"\s*```$", "", txt)
            obj = json.loads(txt)
            names = [str(x).strip() for x in obj.get("candidate_names", []) if str(x).strip()]
            notes = str(obj.get("notes", "") or "")
            if names:
                return names[:5], notes
        except Exception:
            pass

    # heuristic fallback
    q = (query or "").lower()
    avoid_far = ("太遠" in auditor_feedback) or ("distance" in auditor_feedback.lower())
    sorted_shops = sorted(shops, key=lambda s: (float(s.google_rating or 0.0), s.trust_score), reverse=True)
    picked: list[str] = []
    for s in sorted_shops:
        tags_low = {str(t).lower() for t in s.tags}
        if "ramen" in q and "ramen" not in tags_low and "noodle" not in tags_low:
            continue
        if "vegan" in q and "vegan" not in [x.lower() for x in s.allowed_dietary_preferences]:
            continue
        if avoid_far and (s.latitude is None or s.longitude is None):
            continue
        picked.append(s.name)
        if len(picked) >= 4:
            break
    return picked, "heuristic_fallback_researcher"


def node_researcher(state: AgentState) -> AgentState:
    shops = _researcher_shop_pool(state)
    iteration = int(state.get("research_iteration", 0)) + 1
    names, notes = _call_researcher_prompt(
        query=state.get("query", "") or "",
        shops=shops,
        auditor_feedback=state.get("auditor_feedback", "") or "",
        iteration=iteration,
    )
    state["research_iteration"] = iteration
    state["researcher_candidate_names"] = names
    state["researcher_notes"] = notes
    state["transit_audit"].append(
        _dj("researcher_iteration", iteration=iteration, candidate_count=len(names), notes=notes)
    )
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


def node_auditor(state: AgentState) -> AgentState:
    """Professor auditor: physical boundaries + foodie taste."""
    pool = {s.name: s for s in _researcher_shop_pool(state)}
    picked = [n for n in (state.get("researcher_candidate_names") or []) if n in pool]
    if not picked:
        state["auditor_rejected"] = True
        state["auditor_feedback"] = "候選名單是空的，請至少提出 2 家可行店。"
        return state
    reasons: list[str] = []
    now_hm = datetime.now(_APP_TZ).strftime("%H:%M")
    for name in picked:
        s = pool[name]
        if str(s.close_time) <= now_hm:
            reasons.append(f"{name} 現在可能已接近打烊。")
    for a, b in zip(picked, picked[1:]):
        sa, sb = pool[a], pool[b]
        if sa.latitude is not None and sb.latitude is not None and sa.longitude is not None and sb.longitude is not None:
            d = _haversine_km(float(sa.latitude), float(sa.longitude), float(sb.latitude), float(sb.longitude))
            if d > 8.0:
                reasons.append(f"{a} 到 {b} 約 {d:.1f}km，對行程太遠了。")
    avg_rating = sum(float(pool[n].google_rating or 0.0) for n in picked) / max(1, len(picked))
    if avg_rating < 4.1:
        reasons.append("整體口碑偏低，請提高美食家品質門檻。")
    rejected = bool(reasons)
    feedback = "；".join(reasons) if reasons else "審查通過，可進入最終規劃。"
    state["auditor_rejected"] = rejected
    state["auditor_feedback"] = feedback
    state["transit_audit"].append(
        _dj(
            "auditor_review",
            rejected=rejected,
            feedback=feedback,
            candidate_names=picked,
            iteration=state.get("research_iteration", 0),
        )
    )
    return state


def node_audit(state: AgentState) -> AgentState:
    _cleanup_old_txn_logs(retention_days=30)
    txn_saga = SagaEngine(kind="transactional", persist_path=str(_SAGA_DIR / f"saga_txn_{uuid.uuid4().hex}.json"))
    print(_dj("debug_print", node="node_audit", message="Saga transactional reservation"))
    leg1 = state["leg1_offer"]
    leg2 = state["leg2_offer"]
    state["rollback_occurred"] = False

    if not leg1 or not leg2:
        state["rollback_occurred"] = True
        state["transit_audit"].append(
            _dj(
                "flight_offers_missing",
                detail="Flight search returned no results - cannot book.",
            )
        )
        return state

    def reserve_leg1(ctx: dict) -> dict:
        o = ctx["leg1"]
        print(_dj("debug_print", step="reserve_leg1", origin=o["origin"], destination=o["destination"]))
        return {"leg": f"{o['origin']}-{o['destination']}", "price": o["price"]}

    def cancel_leg1(receipt: dict) -> None:
        print(
            _dj(
                "debug_print",
                step="cancel_leg1",
                leg=receipt["leg"],
                refund_usd=receipt["price"],
            )
        )

    def reserve_leg2(ctx: dict) -> dict:
        o = ctx["leg2"]
        print(_dj("debug_print", step="reserve_leg2", origin=o["origin"], destination=o["destination"]))
        if int(o["seats"]) <= 1:
            raise AtomicCommitFailure(
                f"Last-seat race condition on {o['destination']}: offer {o['offer_id']} no longer available."
            )
        return {"leg": f"{o['origin']}-{o['destination']}", "price": o["price"]}

    def cancel_leg2(receipt: dict) -> None:
        print(
            _dj(
                "debug_print",
                step="cancel_leg2",
                leg=receipt.get("leg", "NRT-SFO"),
                refund_usd=receipt.get("price", 0),
            )
        )

    steps = [
        SagaStep("reserve_leg1", reserve_leg1, cancel_leg1),
        SagaStep("reserve_leg2", reserve_leg2, cancel_leg2),
    ]
    ok, log = txn_saga.run(steps, context={"leg1": leg1, "leg2": leg2, "date": "2026-04-24"})

    if not ok:
        failed = next((s for s in log.steps if s.error), None)
        reason = failed.error if failed else "unknown"
        print(_dj("debug_print", level="critical", saga_failed=True, detail=str(reason)))
        state["rollback_occurred"] = True
        state["transit_audit"].append(_dj("saga_audit_failure", detail=str(reason)))
    else:
        total = leg1["price"] + leg2["price"]
        print(
            _dj(
                "debug_print",
                saga_success=True,
                total_usd=round(total, 2),
            )
        )
    return state


def node_plan(state: AgentState) -> AgentState:
    print(_dj("debug_print", node="node_plan", message="Generating outcome report"))
    query_text = state.get("query", "") or ""
    leg1 = state.get("leg1_offer", {})
    leg2 = state.get("leg2_offer", {})

    report = "## Travel Agent - Live Run\n\n"
    report += f"**Run ID:** `{state.get('agent_run_id','')}` "
    report += "(checkpointed by LangGraph SqliteSaver)\n\n"
    if state.get("researcher_candidate_names"):
        report += "### Multi-Agent Loop\n"
        report += f"- Researcher iterations: {int(state.get('research_iteration', 0))}\n"
        report += f"- Candidate draft: {', '.join(state.get('researcher_candidate_names', []))}\n"
        report += f"- Auditor feedback: {state.get('auditor_feedback', '')}\n\n"

    if leg1 and leg2:
        report += "### Flights searched\n"
        report += "| Leg | Flight | Seats | Price |\n|---|---|---|---|\n"
        report += f"| {leg1['origin']}->{leg1['destination']} | {leg1['flight_num']} | {leg1['seats']} | ${leg1['price']} |\n"
        report += f"| {leg2['origin']}->{leg2['destination']} | {leg2['flight_num']} | {leg2['seats']} | ${leg2['price']} |\n\n"

    wants_flight = bool(state.get("wants_flight_search"))

    if state["rollback_occurred"]:
        raw_audit = state["transit_audit"][0] if state["transit_audit"] else ""
        reason = audit_json_line_as_text(raw_audit) if raw_audit else "unknown"
        report += "### TRANSACTION STATUS: ABORTED\n"
        report += f"**Reason:** {reason}\n\n"
    elif leg1 and leg2:
        total = leg1.get("price", 0) + leg2.get("price", 0)
        report += "### TRANSACTION STATUS: SUCCESS\n"
        report += f"Both legs reserved atomically - total **${total:.2f}**.\n\n"
    elif wants_flight:
        report += "### TRANSACTION STATUS: INCOMPLETE\n"
        report += "Flight booking was requested but offers were incomplete or unavailable.\n\n"
    else:
        report += "### Flight booking\n"
        report += "_Not applicable — itinerary planning without flight search._\n\n"

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
    expand_city = _intent.get("city") or "京都"
    expand_region = _intent.get("region") or "jp"
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
        expansion_queries, expansion_mode = _llm_broad_geo_search_queries(
            query_for_city, expand_city, expand_region
        )
        must_for_expansion = sorted(_plan_global_explicit_tags(query_for_city))
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
    if expand_city == "台北":
        seed_shops = list(_build_shop_catalog_taipei())
        state["transit_audit"].append(_dj("seed_shops_loaded", region="tw", profile="taipei", count=len(seed_shops)))
    elif expand_city == "東京":
        seed_shops = list(_build_shop_catalog_tokyo())
        state["transit_audit"].append(_dj("seed_shops_loaded", region="jp", profile="tokyo", count=len(seed_shops)))
    else:
        seed_shops = list(_build_shop_catalog())
        state["transit_audit"].append(_dj("seed_shops_loaded", region="jp", profile="kyoto", count=len(seed_shops)))

    for p in dynamic_pool[:60]:
        prof = _build_dynamic_shop_profile(p, region=str(p.get("region", "jp")))
        dynamic_shops.append(prof)

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

    explicit_category_tags = _plan_global_explicit_tags(query_text)
    must_have_tags = sorted(explicit_category_tags)
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
            s for s in shops if any(str(tag).lower() in must_have_set for tag in s.tags)
        ]
        state["transit_audit"].append(
            _dj(
                "dynamic_plan_filtered_tags",
                tags=must_have_tags,
                kept=len(shops_for_dynamic_planning),
                total=len(shops),
            )
        )

    appetite_light_mode = "appetite_light" in _intent.get("explicit_constraints", [])
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
    phase1_candidates = heuristic_scored[:15]
    phase1_shops = [x.shop for x in phase1_candidates]
    state["transit_audit"].append(
        _dj("hybrid_phase1_heuristic_filter", phase="done", kept=len(phase1_candidates))
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
    requested_slots = list(_intent.get("meal_slots") or [])
    requested_meal_count = _requested_meal_count(state.get("query", ""))
    slot_required_tags = _slot_level_required_tags(state.get("query", ""))
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
    # Phase 2: DP Solver on top-15
    state["transit_audit"].append(_dj("hybrid_phase2_dp_solver", phase="start"))
    graph = GraphBuilder.build_graph(
        ranked=phase1_candidates,
        traffic=traffic_provider,
        start_time=synth_start_time,
        meal_slots=requested_slots,
        mode=mode,
        requested_meal_count=requested_meal_count,
        slot_required_tags=slot_required_tags if slot_required_tags else None,
    )
    desired_len = max(1, requested_meal_count or len(requested_slots or phase1_candidates[:3]))
    k_paths = ItinerarySynthesizer.find_k_optimal_paths(
        graph=graph,
        required_length=desired_len,
        must_have_tags=explicit_category_tags,
        k=5,
    )
    state["transit_audit"].append(
        _dj("hybrid_phase2_dp_solver", phase="done", paths=len(k_paths))
    )

    # Phase 3: Saga Commitment (SNS probe + flight lock), fallback to next-best path on failure.
    state["transit_audit"].append(_dj("hybrid_phase3_saga_commitment", phase="start"))
    ranked_by_name = {r.shop.name: r for r in phase1_candidates}
    selected_ranked_path: list[RankedShop] = []
    risk_keywords = ("火山", "臨休", "休業", "完売", "sold out")
    for idx, path in enumerate(k_paths, start=1):
        path_names = [n.shop_name for n in path]
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
        if not leg1 or not leg2:
            state["transit_audit"].append(
                _dj("hybrid_phase3_fail", path_index=idx, reason="flight_lock_missing")
            )
            phase3_failed = True
        if phase3_failed:
            continue
        selected_ranked_path = [ranked_by_name[n] for n in path_names if n in ranked_by_name]
        state["transit_audit"].append(
            _dj("hybrid_phase3_commit", path_index=idx, shops=path_names)
        )
        break
    if not selected_ranked_path:
        selected_ranked_path = ranked[: max(1, min(3, len(ranked)))]
        state["transit_audit"].append(
            _dj(
                "hybrid_phase3_fallback",
                detail="no committed path; using heuristic top picks",
            )
        )

    ranked = selected_ranked_path
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
    )
    boundary_skips = [w for w in synthesized.warnings if w.startswith("OPERATING_BOUNDARY_SKIP")]
    for skip_msg in boundary_skips:
        state["transit_audit"].append(
            _dj("synthesis_warning_forwarded", warning=skip_msg)
        )
    if synthesized.rollback_triggered:
        state["rollback_occurred"] = True
        state["transit_audit"].append(_dj("early_interception_rollback_triggered"))

    # Health_Check before commitment: if over budget, rollback to healthier backup.
    if ranked:
        top_shop = ranked[0].shop
        projected_health = health_tracker.spent + ScoringEngine.optimized_health_impact(top_shop)
        if projected_health > user_pref.health_budget_limit:
            fallback = choose_health_backup([r.shop for r in ranked], top_shop)
            state["rollback_occurred"] = True
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


def node_collect_feedback(state: AgentState) -> AgentState:
    global _learned_weight_profile, _feedback_samples_seen, _feedback_penalties, _taste_max_blacklist
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


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("route_intent", node_route_intent)
    g.add_node("flight_search", node_flight_search)
    g.add_node("food_search", node_food_search)
    g.add_node("researcher", node_researcher)
    g.add_node("auditor", node_auditor)
    g.add_node("audit", node_audit)
    g.add_node("collect_feedback", node_collect_feedback)
    g.add_node("plan", node_plan)
    g.set_entry_point("route_intent")

    def _branch_after_intent(state: AgentState) -> str:
        return "flight_search" if state.get("wants_flight_search") else "food_search"

    g.add_conditional_edges(
        "route_intent",
        _branch_after_intent,
        {"flight_search": "flight_search", "food_search": "food_search"},
    )
    g.add_edge("flight_search", "food_search")
    g.add_edge("food_search", "researcher")
    g.add_edge("researcher", "auditor")

    def _after_auditor(state: AgentState) -> str:
        rejected = bool(state.get("auditor_rejected"))
        iteration = int(state.get("research_iteration", 0))
        if rejected and iteration < 3:
            return "researcher"
        if (state.get("intent") or {}).get("mode") == "right_now":
            return "plan"
        if state.get("wants_flight_search"):
            return "audit"
        return "collect_feedback"

    g.add_conditional_edges(
        "auditor",
        _after_auditor,
        {"researcher": "researcher", "plan": "plan", "audit": "audit", "collect_feedback": "collect_feedback"},
    )
    g.add_edge("audit", "collect_feedback")
    g.add_edge("collect_feedback", "plan")
    g.add_edge("plan", END)
    if SqliteSaver is not None:
        cp = SqliteSaver.from_conn_string(str(_SAGA_DIR / "graph_checkpoints.sqlite"))
        return g.compile(checkpointer=cp)
    return g.compile()


if __name__ == "__main__":
    initial_state: AgentState = make_initial_state("Book TPE to SFO via NRT")
    result = build_graph().invoke(initial_state)
    print(_dj("cli_demo_done", itinerary_chars=len(result.get("final_itinerary", "") or "")))
    print(result["final_itinerary"])
