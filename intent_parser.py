"""
Hybrid intent parser: rule fast-path → LLM fallback (+ optional refinement path).

Public API
----------
parse_intent(query, llm_router, *, user_locale, user_lat, user_lng,
             previous_intent=None, prev_itinerary=None)                     -> Intent
intent_from_snapshot_dict(d)                                      -> Intent
parse_intent_rules(query, *, user_locale, user_lat, user_lng)     -> Intent | None
parse_intent_llm(query, llm_router, *, previous_intent=None,
                  prev_itinerary=None)                            -> Intent

Intent dataclass fields
-----------------------
city               str | None   concrete city name; ``None`` when unknown (never silently defaulted)
region             str          "tw" | "jp" | "unknown"
meal_slots         list[str]    subset of breakfast/lunch/tea/dinner/late_night
time_window        tuple[str|None, str|None]  (HH:MM start, HH:MM end) or Nones
category_tags      list[str]    e.g. ["ramen", "dessert"]
dietary_hints      str | None   "vegan" | "vegetarian" | "pescatarian" | "no_beef" | "no_ramen" | "no_pork" | None
excluded_shops     list[str]    venue names to avoid (「不想吃 X」); prefer canonical storefront strings
excluded_tags      list[str]    category / cuisine tags to avoid (e.g. matcha, cafe); not venue names
mode               str          "right_now" | "balanced" | "taste_max"
explicit_constraints list[str]  e.g. ["appetite_light", "strong_ramen"]
wants_flight       bool
confidence         float        0.0–1.0 (rules estimate)
is_revision        bool         refinement-turn marker (typically from LLM when prior exists)
is_actionable      bool         False → route_intent ends early with LLM ``actionability_followup``
actionability_followup str|None  User-facing (A)–(D) clarification when not actionable

Design notes
------------
* Rule helpers are private (_-prefixed) and live here; agent.py no longer
  imports them directly.
* LLM output is validated with pydantic; a single retry is attempted on
  schema mismatch, with OTEL span attributes recorded.
* With ``previous_intent`` set (multi-turn), ``parse_intent`` runs ONE targeted
  refinement LLM call first (minimal re-tokenisation vs cold extraction).
  Otherwise it skips LLM entirely when rule confidence ≥ 0.6.
"""

from __future__ import annotations

import copy
from datetime import datetime
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Global schedule (mock) for slot collision detection
# ---------------------------------------------------------------------------
_GLOBAL_SCHEDULE: dict[str, list[str]] = {"dinner": ["yakiniku"]}

def set_global_schedule(schedule: dict[str, str]) -> None:
    """Set the global schedule (e.g., {"lunch": "拉麵", "dinner": "燒肉"})."""
    global _GLOBAL_SCHEDULE
    _GLOBAL_SCHEDULE = dict(schedule)

def get_global_schedule() -> dict[str, str]:
    """Return a copy of the current global schedule."""
    return dict(_GLOBAL_SCHEDULE)

from pydantic import BaseModel, field_validator, model_validator

from observability import record_llm_call, _get_tracer
from shop_planning import ShopProfile


def _strip_city_optional(raw: str | None) -> str | None:
    """Normalize intent city: empty string → None (never invent a default city)."""
    if raw is None:
        return None
    s = str(raw).strip()
    return None if s == "" else s

# ---------------------------------------------------------------------------
# Intent data structure
# ---------------------------------------------------------------------------

@dataclass
class RevisionOp:
    op_type: str          # "replace" | "remove" | "insert" | "swap"
    target_shop: str      # 被操作的店名（replace/remove/swap 用）
    new_shop: str | None  # replace/swap 的替換目標；其餘為 None
    slot_id: str | None   # 對應 stable UUID；初始為 None，由 agent 填入


@dataclass
class Intent:
    city: str | None = None
    region: str = "unknown"
    meal_slots: list[str] = field(default_factory=list)
    time_window: tuple[str | None, str | None] = (None, None)
    category_tags: list[str] = field(default_factory=list)
    dietary_hints: str | None = None
    excluded_shops: list[str] = field(default_factory=list)
    excluded_tags: list[str] = field(default_factory=list)
    #: Hard lock: shops that MUST appear in the final DP itinerary.
    must_include_shops: list[str] = field(default_factory=list)
    #: Hard lock: shops that MUST be excluded from the final DP itinerary.
    must_exclude_shops: list[str] = field(default_factory=list)
    mode: str = "balanced"
    explicit_constraints: list[str] = field(default_factory=list)
    wants_flight: bool = False
    confidence: float = 0.0
    revision_op: RevisionOp | None = None
    confirm_op: dict | None = None
    #: {"message": "..."} when user says 「幫我存檔」
    #: True when this intent updates a stored prior snapshot (refinement turn).
    is_revision: bool = False
    #: False when the query is too vague to run retrieval/planning without clarification.
    is_actionable: bool = True
    #: User-facing follow-up when ``is_actionable`` is false; must include (A)(B)(C)(D) options.
    actionability_followup: str | None = None
    #: Additional metadata about intent source
    #: Pending mutation awaiting user confirmation (e.g., meal_slots change).
    pending_mutation: dict | None = None
    #: Pending replacement awaiting user confirmation (slot collision with global schedule).
    pending_replacement: dict | None = None
    metadata: dict = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation for AgentState storage."""
        rev_op = None
        if self.revision_op is not None:
            rev_op = {
                "op_type": self.revision_op.op_type,
                "target_shop": self.revision_op.target_shop,
                "new_shop": self.revision_op.new_shop,
                "slot_id": self.revision_op.slot_id,
            }
        return {
            "city": self.city,
            "region": self.region,
            "meal_slots": list(self.meal_slots),
            "time_window": list(self.time_window),
            "category_tags": list(self.category_tags),
            "dietary_hints": self.dietary_hints,
            "excluded_shops": list(self.excluded_shops),
            "excluded_tags": list(self.excluded_tags),
            "must_include_shops": list(self.must_include_shops),
            "must_exclude_shops": list(self.must_exclude_shops),
            "mode": self.mode,
            "explicit_constraints": list(self.explicit_constraints),
            "wants_flight": self.wants_flight,
            "confidence": self.confidence,
            "is_revision": self.is_revision,
            "is_actionable": self.is_actionable,
            "actionability_followup": self.actionability_followup,
            "pending_mutation": self.pending_mutation,
            "pending_replacement": self.pending_replacement,
            "revision_op": rev_op,
            "confirm_op": self.confirm_op,
            "metadata": self.metadata,
        }


def intent_from_snapshot_dict(d: dict[str, Any]) -> Intent:
    """Rebuild :class:`Intent` from ``AgentState['intent']`` / :meth:`Intent.as_dict` output."""
    raw_tw = d.get("time_window")
    if isinstance(raw_tw, dict):
        tw: tuple[str | None, str | None] = (raw_tw.get("start"), raw_tw.get("end"))
    elif isinstance(raw_tw, (list, tuple)) and len(raw_tw) >= 2:
        tw = (raw_tw[0], raw_tw[1])
    elif isinstance(raw_tw, (list, tuple)) and len(raw_tw) == 1:
        tw = (raw_tw[0], None)
    else:
        tw = (None, None)
    rc = d.get("city")
    rev_op_raw = d.get("revision_op")
    rev_op_instance = None
    if rev_op_raw is not None and isinstance(rev_op_raw, dict):
        rev_op_instance = RevisionOp(
            op_type=str(rev_op_raw.get("op_type")),
            target_shop=str(rev_op_raw.get("target_shop")),
            new_shop=rev_op_raw.get("new_shop"),
            slot_id=rev_op_raw.get("slot_id"),
        )
    return Intent(
        city=_strip_city_optional(str(rc) if rc is not None else None),
        region=str(d.get("region") or "unknown"),
        meal_slots=[str(x) for x in (d.get("meal_slots") or []) if x is not None],
        time_window=tw,
        category_tags=[str(x) for x in (d.get("category_tags") or []) if x is not None],
        dietary_hints=d.get("dietary_hints"),
        excluded_shops=[str(x) for x in (d.get("excluded_shops") or []) if x is not None],
        excluded_tags=list(
            dict.fromkeys(
                str(x).strip().lower()
                for x in (d.get("excluded_tags") or [])
                if x is not None and len(str(x).strip()) >= 2
            )
        ),
        must_include_shops=[str(x) for x in (d.get("must_include_shops") or []) if x is not None],
        must_exclude_shops=[str(x) for x in (d.get("must_exclude_shops") or []) if x is not None],
        mode=str(d.get("mode") or "balanced"),
        explicit_constraints=[str(x) for x in (d.get("explicit_constraints") or []) if x is not None],
        wants_flight=bool(d.get("wants_flight", False)),
        confidence=float(d.get("confidence", 0.0)),
        is_revision=bool(d.get("is_revision", False)),
        is_actionable=bool(d.get("is_actionable", True)),
        actionability_followup=(
            None
            if d.get("actionability_followup") in (None, "")
            else str(d.get("actionability_followup")).strip() or None
        ),
        pending_mutation=d.get("pending_mutation"),
        pending_replacement=d.get("pending_replacement"),
        revision_op=rev_op_instance,
        confirm_op=d.get("confirm_op"),
        metadata=dict(d.get("metadata", {})),
    )


# ---------------------------------------------------------------------------
# Private rule helpers (kept in this module; agent.py should not import them)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _TimeRange:
    start: str | None = None
    end: str | None = None


def _clamp_hhmm(token: str) -> str:
    parts = token.split(":", 1)
    h = max(0, min(23, int(parts[0])))
    m = max(0, min(59, int(parts[1]) if len(parts) > 1 else 0))
    return f"{h:02d}:{m:02d}"


def _normalize_hhmm(hour: int, minute: int, period: str | None) -> str:
    h = max(0, min(23, int(hour)))
    m = max(0, min(59, int(minute)))
    if period:
        p = period.lower()
        if p in {"下午", "晚上", "pm", "p.m.", "傍晚"} and h < 12:
            h += 12
        if p == "凌晨" and h == 12:
            h = 0
    return f"{h:02d}:{m:02d}"


def _extract_time_range(query: str) -> _TimeRange:
    text = query or ""
    m = re.search(r"(\d{1,2}:\d{2})\s*[-–~～到至]\s*(\d{1,2}:\d{2})", text)
    if m:
        return _TimeRange(start=_clamp_hhmm(m.group(1)), end=_clamp_hhmm(m.group(2)))
    m2 = re.search(
        r"(早上|上午|中午|下午|晚上|凌晨|pm|am)?\s*(\d{1,2})(?::(\d{2}))?\s*點?\s*[-–~～到至]\s*"
        r"(早上|上午|中午|下午|晚上|凌晨|pm|am)?\s*(\d{1,2})(?::(\d{2}))?\s*點?",
        text,
        flags=re.IGNORECASE,
    )
    if m2:
        p1, h1, mn1, p2, h2, mn2 = m2.group(1), m2.group(2), m2.group(3), m2.group(4), m2.group(5), m2.group(6)
        return _TimeRange(
            start=_normalize_hhmm(int(h1), int(mn1 or "0"), p1),
            end=_normalize_hhmm(int(h2), int(mn2 or "0"), p2),
        )
    return _TimeRange()


def _is_right_now_mode(query: str) -> bool:
    q = (query or "").upper()
    return ("RIGHT_NOW" in q) or ("現在餓了" in query) or ("附近 30 分鐘" in query)


def _is_taipei_query(query: str) -> bool:
    q = (query or "").upper()
    return ("台北" in query) or ("TAIPEI" in q) or ("台灣" in query) or ("TAIWAN" in q)


def _is_tokyo_query(query: str) -> bool:
    q = (query or "").upper()
    return ("東京" in query) or ("TOKYO" in q)


def _is_flight_intent(query: str) -> bool:
    q = query or ""
    low = q.lower()
    return ("機票" in q) or ("flight" in low) or ("book" in low)


def _is_ramen(query: str) -> bool:
    return any(k in (query or "").lower() for k in ("拉麵", "拉面", "ramen"))


def _is_appetite_light(query: str) -> bool:
    return any(k in (query or "") for k in ("吃不太下", "吃不下", "食量小", "小食量", "胃口小"))


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
        mo = re.search(pat, q)
        if not mo:
            continue
        token = mo.group(1)
        if token.isdigit():
            return max(1, min(5, int(token)))
        if token in cn_map:
            return cn_map[token]
    return None


def _requested_meal_slots(query: str) -> list[str]:
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

    tr = _extract_time_range(query)
    if tr.start:
        hh = int(tr.start.split(":")[0])
        if hh <= 10 and "breakfast" not in slots:
            slots = ["breakfast", *slots]

    seen: set[str] = set()
    deduped: list[str] = []
    for s in slots:
        if s not in seen:
            deduped.append(s)
            seen.add(s)
    slots = deduped

    if requested_count is not None:
        target = max(1, min(5, requested_count))
        morning_hint = bool(tr.start and int(tr.start.split(":")[0]) <= 10) or any(
            k in q for k in ("早上", "清晨", "早餐", "morning", "breakfast")
        )
        ramen_all = _is_ramen(query) and target == 3 and morning_hint
        if ramen_all:
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


def _extract_category_tags(query: str) -> set[str]:
    q = (query or "").lower()
    tags: set[str] = set()
    if _is_ramen(query):
        tags.add("ramen")
    if any(k in q for k in ("居酒屋", "izakaya")):
        tags.add("izakaya")
    if any(k in q for k in ("串燒", "串烧", "燒鳥", "烧鸟", "焼き鳥", "yakitori")):
        tags.add("izakaya")
    if any(k in q for k in ("蛋糕", "甜點", "甜点", "下午茶", "巴斯克", "提拉米蘇", "抹茶")):
        tags.add("dessert")
    if any(k in q for k in ("壽司", "寿司", "sushi")):
        tags.add("sushi")
    if any(k in q for k in ("燒肉", "烧肉", "烤肉", "yakiniku")):
        tags.add("yakiniku")
    return tags


def _dietary_ethics(query: str) -> str | None:
    q_raw = query or ""
    q = q_raw.lower()
    if "vegan" in q or "純素" in q_raw:
        return "vegan"
    if "vegetarian" in q or "素食" in q_raw:
        return "vegetarian"
    if "pescatarian" in q:
        return "pescatarian"
    if any(k in q_raw for k in ("不吃牛", "不吃牛肉")) or "no beef" in q or "avoid beef" in q:
        return "no_beef"
    if any(k in q_raw for k in ("不吃拉麵", "不吃拉面")) or "no ramen" in q or "avoid ramen" in q:
        return "no_ramen"
    if any(k in q_raw for k in ("不吃豬", "不吃猪", "不吃豚")) or "no pork" in q or "avoid pork" in q:
        return "no_pork"
    return None


_RE_EXCLUDED_SHOP_CHUNK = re.compile(
    r"(?:不想吃|不要吃|不喜歡吃|忌口|避開|排除)(?:這家|那家|本店)?\s*[：:]?\s*"
    r"([\u3040-\u30ff\u4e00-\u9fffA-Za-z0-9〇・．.·\-＆&/／、，, ]{2,48})",
    flags=re.UNICODE,
)


def _extract_excluded_shops_quick(query: str) -> list[str]:
    """Surface-form venue tokens; LLM path should normalize to canonical storefront names."""
    text = query or ""
    out: list[str] = []
    for m in _RE_EXCLUDED_SHOP_CHUNK.finditer(text):
        chunk = m.group(1).strip()
        if not chunk:
            continue
        for sep in ("、", ",", "，", "/", "／"):
            if sep in chunk:
                for part in chunk.split(sep):
                    p = part.strip().rstrip("的店館院所")
                    if len(p) >= 2:
                        out.append(p)
                break
        else:
            chunk = chunk.rstrip("的店館院所")
            if len(chunk) >= 2:
                out.append(chunk)
    seen: set[str] = set()
    deduped: list[str] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            deduped.append(x)
    return deduped


def _locale_to_city_region(token: str) -> tuple[str, str] | None:
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


def _coords_to_city_region(lat: float | None, lng: float | None) -> tuple[str, str] | None:
    if lat is None or lng is None:
        return None
    if 24.8 <= lat <= 25.3 and 121.3 <= lng <= 121.8:
        return ("台北", "tw")
    if 35.4 <= lat <= 36.0 and 139.4 <= lng <= 140.1:
        return ("東京", "jp")
    return None


def _resolve_city(
    query: str,
    user_locale: str | None = None,
    user_lat: float | None = None,
    user_lng: float | None = None,
) -> tuple[str, str, bool]:
    """Return (city, region, city_in_query)."""
    q_lower = (query or "").lower()
    q_raw = query or ""
    if ("台北" in q_raw) or ("taipei" in q_lower) or ("台灣" in q_raw) or ("taiwan" in q_lower):
        return ("台北", "tw", True)
    if ("京都" in q_raw) or ("kyoto" in q_lower):
        return ("京都", "jp", True)
    if ("東京" in q_raw) or ("tokyo" in q_lower):
        return ("東京", "jp", True)
    if ("大阪" in q_raw) or ("osaka" in q_lower):
        return ("大阪", "jp", True)

    ul = _locale_to_city_region(user_locale or "")
    if ul is not None:
        return (*ul, False)

    coord = _coords_to_city_region(user_lat, user_lng)
    if coord is not None:
        return (*coord, False)

    env_hint = os.getenv("DEFAULT_LOCALE_CITY", "").strip()
    env_mapped = _locale_to_city_region(env_hint)
    if env_mapped is not None:
        return (*env_mapped, False)

    return (None, "unknown", False)


# ---------------------------------------------------------------------------
# Pydantic schema for LLM output validation
# ---------------------------------------------------------------------------

_VALID_SLOTS = {"breakfast", "lunch", "tea", "dinner", "late_night"}
_VALID_MODES = {"right_now", "balanced", "taste_max"}
_VALID_REGIONS = {"tw", "jp", "unknown"}

_DEFAULT_MISSING_CITY_FOLLOWUP = (
    "請指定要規劃的城市（無法從您的描述推斷）：\n"
    "(A) 東京\n(B) 大阪\n(C) 京都\n(D) 台北"
)


class _LLMIntentSchema(BaseModel):
    city: str | None = None
    region: str = "unknown"
    meal_slots: list[str] = []
    time_window: dict[str, str | None] = {"start": None, "end": None}
    category_tags: list[str] = []
    dietary_hints: str | None = None
    excluded_shops: list[str] = []
    excluded_tags: list[str] = []
    must_include_shops: list[str] = []
    must_exclude_shops: list[str] = []
    mode: str = "balanced"
    explicit_constraints: list[str] = []
    wants_flight: bool = False
    confidence: float = 0.8
    is_revision: bool = False
    is_actionable: bool = True
    actionability_followup: str | None = None
    revision_op: dict | None = None
    confirm_op: dict | None = None

    @field_validator("city", mode="before")
    @classmethod
    def _normalize_city_nullable(cls, v: object) -> str | None:
        if v is None:
            return None
        if isinstance(v, str):
            s = v.strip()
            return None if s == "" else s
        s = str(v).strip()
        return None if s == "" else s

    @field_validator("excluded_shops", mode="before")
    @classmethod
    def _normalize_excluded_shops(cls, v: object) -> list[str]:
        if v is None:
            return []
        if not isinstance(v, list):
            return []
        seen: set[str] = set()
        out: list[str] = []
        for x in v:
            if x is None:
                continue
            s = str(x).strip()
            if len(s) < 2:
                continue
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out

    @field_validator("excluded_tags", mode="before")
    @classmethod
    def _normalize_excluded_tags(cls, v: object) -> list[str]:
        if v is None:
            return []
        if not isinstance(v, list):
            return []
        seen: set[str] = set()
        out: list[str] = []
        for x in v:
            if x is None:
                continue
            s = str(x).strip().lower()
            if len(s) < 2:
                continue
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out

    @field_validator("meal_slots")
    @classmethod
    def _validate_slots(cls, v: list[str]) -> list[str]:
        return [s for s in v if s in _VALID_SLOTS]

    @field_validator("mode")
    @classmethod
    def _validate_mode(cls, v: str) -> str:
        return v if v in _VALID_MODES else "balanced"

    @field_validator("region")
    @classmethod
    def _validate_region(cls, v: str) -> str:
        return v if v in _VALID_REGIONS else "unknown"

    @field_validator("confidence")
    @classmethod
    def _clamp_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))

    @model_validator(mode="after")
    def _coerce_time_window(self) -> "_LLMIntentSchema":
        if not isinstance(self.time_window, dict):
            self.time_window = {"start": None, "end": None}
        return self


def _parse_revision_op(raw: dict | None) -> RevisionOp | None:
    if not raw or not isinstance(raw, dict):
        return None
    op_type = raw.get("op_type")
    if not op_type:
        return None
    return RevisionOp(
        op_type=str(op_type),
        target_shop=str(raw.get("target_shop") or ""),
        new_shop=raw.get("new_shop"),
        slot_id=raw.get("slot_id"),
    )


def _schema_to_intent(s: _LLMIntentSchema) -> Intent:
    tw = s.time_window or {}
    actionable = bool(s.is_actionable)
    fu = (s.actionability_followup or "").strip()
    if not actionable and not fu:
        fu = "請告訴我您具體想去的城市或國家，以便我為您規劃。"
    city_val = _strip_city_optional(s.city)
    reg = str(s.region or "unknown")
    if reg not in _VALID_REGIONS:
        reg = "unknown"
    return Intent(
        city=city_val,
        region=reg,
        meal_slots=list(s.meal_slots),
        time_window=(tw.get("start"), tw.get("end")),
        category_tags=list(s.category_tags),
        dietary_hints=s.dietary_hints,
        excluded_shops=list(s.excluded_shops),
        excluded_tags=list(s.excluded_tags),
        must_include_shops=list(s.must_include_shops),
        must_exclude_shops=list(s.must_exclude_shops),
        mode=str(s.mode),
        explicit_constraints=list(s.explicit_constraints),
        wants_flight=bool(s.wants_flight),
        confidence=float(s.confidence),
        is_revision=bool(s.is_revision),
        is_actionable=actionable,
        actionability_followup=fu if fu else None,
        revision_op=_parse_revision_op(s.revision_op),
        confirm_op=s.confirm_op,
    )


def _trip_requires_resolved_city(intent: Intent) -> bool:
    """Whether this intent implies needing a concrete city before search/plan (vs pure dietary text)."""
    if intent.wants_flight:
        return True
    if intent.meal_slots:
        return True
    if intent.category_tags:
        return True
    if intent.mode == "right_now":
        return True
    if intent.explicit_constraints:
        return True
    if intent.dietary_hints or intent.excluded_tags:
        return False
    return True


def _clamp_missing_city_if_actionable(intent: Intent) -> None:
    """After geo pins: trip-like intents must not stay actionable without a resolved city."""
    intent.city = _strip_city_optional(intent.city)
    if not intent.is_actionable:
        return
    if not _trip_requires_resolved_city(intent):
        return
    if intent.city:
        return
    intent.is_actionable = False
    if not (intent.actionability_followup or "").strip():
        intent.actionability_followup = _DEFAULT_MISSING_CITY_FOLLOWUP


def _llm_clarification_strategy(
    query: str,
    previous: "Intent",
    new: "Intent",
    removed: list[str],
    added: list[str],
    llm_router: Any,
    itinerary_slots: list[dict] | None = None,
) -> tuple[str, str]:
    """
    Ask LLM to decide clarification strategy for meal_slots conflict.
    Returns (strategy, followup_question).
    strategy: "auto_fix" | "ask" | "disambiguate"
    followup_question: empty string if auto_fix
    """
    MEAL_LABEL = {
        "breakfast": "早餐",
        "lunch": "午餐",
        "tea": "下午茶",
        "dinner": "晚餐",
        "late_night": "宵夜",
    }
    removed_str = "、".join(MEAL_LABEL.get(s, s) for s in removed)
    added_str = "、".join(MEAL_LABEL.get(s, s) for s in added)
    prev_str = "、".join(MEAL_LABEL.get(s, s) for s in previous.meal_slots)
    new_str = "、".join(MEAL_LABEL.get(s, s) for s in new.meal_slots)

    # 建立實際行程描述
    if itinerary_slots:
        slot_desc = "、".join(
            f"{MEAL_LABEL.get(s.get('meal_type',''), s.get('meal_type',''))}：{s.get('shop_name','（未安排）')}"
            for s in itinerary_slots
        )
    else:
        slot_desc = "（尚無行程）"

    prompt = f"""你是旅遊規劃助手的 State Manager。

使用者說：「{query}」
前一輪設定的餐次：{prev_str}
LLM 解析出的新餐次：{new_str}
消失的餐次：{removed_str if removed_str else "無"}
新增的餐次：{added_str if added_str else "無"}
目前實際已排行程：{slot_desc}

注意：「前一輪設定的餐次」是 intent 的設定，不代表每個餐次都已經有餐廳。
請根據「目前實際已排行程」判斷哪些餐次真的有被安排，哪些還是空的。

判斷這個差異的原因，並選擇處理策略：

A. auto_fix：LLM 自己解析時漏掉了（False Negative），
   使用者沒有要求取消，直接恢復原本的行程，不問使用者。
   適用：使用者的 query 完全沒有提到任何餐次，只是換店或改 mode。
   注意：如果使用者只提到部分餐次（例如只說「換晚餐」），
   其他沒有被提到的餐次狀態是模糊的，不應該 auto_fix，應該用 ask。

B. ask：需要向使用者確認一件事（是非題）。
   適用：使用者的 query 涉及到消失的餐次，但不確定是要換還是取消。

C. disambiguate：使用者意圖有歧義，需要給選項（單選題）。
   適用：無法從 query 判斷使用者真正的意圖。

只回傳 JSON，格式如下，不要其他文字：
{{"strategy": "auto_fix"}}
或
{{"strategy": "ask", "question": "（繁體中文問題，結尾給 (A)(B) 選項）"}}
或
{{"strategy": "disambiguate", "question": "（繁體中文問題，結尾給 (A)(B)(C) 選項）"}}
"""
    messages = [{"role": "user", "content": prompt}]
    try:
        from llm_router import TaskType  # local import to avoid circular deps at module load
        response = llm_router.complete(TaskType.INTENT_PARSING, messages)
        raw = (response.content or "").strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
        print(f"[Clarification LLM] raw response: {raw!r}")
        data = json.loads(raw)
        strategy = data.get("strategy", "ask")
        question = data.get("question", "")
        if strategy not in ("auto_fix", "ask", "disambiguate"):
            strategy = "ask"
        return strategy, question
    except Exception:
        # fallback: 保守地問使用者
        removed_label = "、".join(MEAL_LABEL.get(s, s) for s in removed)
        fallback_q = (
            f"我注意到【{removed_label}】從行程中消失了。\n"
            f"(A) 對，取消【{removed_label}】\n"
            f"(B) 不，請保留【{removed_label}】"
        )
        return "ask", fallback_q


def _extract_shop_lock_ops(
    query: str,
    itinerary_slots: list[dict] | None,
    intent: "Intent",
) -> None:
    """
    Rule-based post-processor: 從 query 抽取明確的 must_include / must_exclude 指令。
    直接修改 intent（in-place）。
    
    Patterns:
      must_include: 「XX 不要動」「不要動 XX」「保留 XX」「keep XX」
      must_exclude: 「把 XX 換掉」「換掉 XX」「不要 XX（店名）」
    """
    if not query or not itinerary_slots:
        return

    q = query.strip()
    shop_names = [s.get("shop_name", "") for s in itinerary_slots if s.get("shop_name")]
    if not shop_names:
        return

    must_include: list[str] = list(intent.must_include_shops or [])
    must_exclude: list[str] = list(intent.must_exclude_shops or [])

    for shop in shop_names:
        if not shop:
            continue
        # must_include patterns
        if (
            f"{shop}不要動" in q
            or f"{shop} 不要動" in q
            or f"不要動{shop}" in q
            or f"不要動 {shop}" in q
            or f"保留{shop}" in q
            or f"保留 {shop}" in q
            or f"keep {shop}" in q.lower()
        ):
            if shop not in must_include:
                must_include.append(shop)

        # must_exclude patterns（只在店名前面有明確換掉指令）
        if (
            f"把{shop}換掉" in q
            or f"把 {shop} 換掉" in q
            or f"換掉{shop}" in q
            or f"換掉 {shop}" in q
            or f"不要{shop}" in q
            or f"不要 {shop}" in q
        ):
            if shop not in must_exclude:
                must_exclude.append(shop)
            # 如果同時在 must_include，移除（矛盾時以 exclude 優先）
            if shop in must_include:
                must_include.remove(shop)

    intent.must_include_shops = must_include
    intent.must_exclude_shops = must_exclude


def _reconcile_intents(
    previous: Intent,
    new: Intent,
    global_schedule: dict | None = None,
    query: str = "",
    llm_router: Any = None,
    itinerary_slots: list[dict] | None = None,
) -> Intent:
    """Merge new revision intent into previous, applying state reconciliation rules."""
    if not new.is_revision:
        return new

    # Start from a copy of previous
    merged = copy.deepcopy(previous)

    # If there is already a pending mutation or pending replacement, do not apply any meal_slots changes
    # (the user must confirm or reject the pending change first).
    if previous.pending_mutation is not None or previous.pending_replacement is not None:
        # Keep the pending state; ignore any meal_slots from new.
        # Still merge other fields as usual.
        pass
    else:
        # Detect meal_slots conflict and create pending mutation
        if new.meal_slots and previous.meal_slots and new.meal_slots != previous.meal_slots:
            removed = [s for s in previous.meal_slots if s not in new.meal_slots]
            added = [s for s in new.meal_slots if s not in previous.meal_slots]
            print(f"[State Manager] Detected meal_slots diff: removed={removed}, added={added}")

            if llm_router and (removed or added):
                strategy, followup = _llm_clarification_strategy(
                    query, previous, new, removed, added, llm_router,
                    itinerary_slots=itinerary_slots,
                )
            else:
                # fallback without LLM
                strategy = "ask"
                MEAL_LABEL = {"breakfast":"早餐","lunch":"午餐","tea":"下午茶","dinner":"晚餐","late_night":"宵夜"}
                removed_label = "、".join(MEAL_LABEL.get(s, s) for s in removed)
                followup = (
                    f"我注意到【{removed_label}】從行程中消失了。\n"
                    f"(A) 對，取消【{removed_label}】\n"
                    f"(B) 不，請保留【{removed_label}】"
                ) if removed else ""

            if strategy == "auto_fix":
                # False Negative：直接恢復，不問使用者
                merged.meal_slots = list(previous.meal_slots)
            else:
                # ask or disambiguate：問使用者
                merged.is_actionable = False
                merged.pending_mutation = {"meal_slots": list(new.meal_slots)}
                merged.actionability_followup = followup
                return merged
        else:
            # Overwrite meal_slots if changed (no conflict)
            if new.meal_slots and new.meal_slots != previous.meal_slots:
                print(f"[State Manager] Detected meal_slots overwrite: {previous.meal_slots} -> {new.meal_slots}")
                merged.meal_slots = list(new.meal_slots)

    # After merging meal_slots, check for slot collisions with global schedule
    # (only if no pending mutation/replacement already)
    if merged.pending_mutation is None and merged.pending_replacement is None:
        merged = _check_global_schedule_collision(merged, global_schedule=global_schedule)
        if merged.pending_replacement is not None:
            # Collision was detected; return early without applying other changes.
            return merged

        # Check for category_tags conflict (unreasonable expansion)
        old_tags = set(previous.category_tags)
        new_tags = set(new.category_tags)
        if old_tags and not old_tags.issuperset(new_tags):
            added_tags = list(new_tags - old_tags)
            print(f"[State Manager] Detected category_tags conflict: old={old_tags}, new={new_tags}, added={added_tags}")
            merged.is_actionable = False
            merged.pending_replacement = {
                "added_tags": added_tags,
                "old_tags": list(old_tags),
            }
            old_str = "、".join(old_tags)
            new_str = "、".join(added_tags)
            merged.actionability_followup = (
                f"您稍早想安排【{old_str}】，現在又提到【{new_str}】。\n"
                f"請問您打算怎麼安排呢？\n"
                f"(A) 換成【{new_str}】（取消原本的）\n"
                f"(B) 維持【{old_str}】（忽略這次的新增）\n"
                f"(C) 兩個都吃！把【{new_str}】排在【{old_str}】之前\n"
                f"(D) 兩個都吃！把【{new_str}】排在【{old_str}】之後"
            )
            # Restore category_tags to the old tags (keep state clean).
            merged.category_tags = list(old_tags)
            return merged

    # Merge dietary_hints (prefer new if non-null, else keep previous)
    if new.dietary_hints is not None:
        merged.dietary_hints = new.dietary_hints

    # Merge excluded_tags (union)
    if new.excluded_tags:
        merged.excluded_tags = list(dict.fromkeys(previous.excluded_tags + new.excluded_tags))

    # Merge excluded_shops (union)
    if new.excluded_shops:
        merged.excluded_shops = list(dict.fromkeys(previous.excluded_shops + new.excluded_shops))

    # Merge category_tags (union)
    if new.category_tags:
        merged.category_tags = list(dict.fromkeys(previous.category_tags + new.category_tags))

    # Override city if new provides a non-null city
    if new.city is not None:
        merged.city = new.city

    # Override region if new provides a non-null region
    if new.region != "unknown":
        merged.region = new.region

    # Override mode if new provides a non-default mode
    if new.mode != "balanced":
        merged.mode = new.mode

    # Override time_window if new provides non-null start/end
    if new.time_window != (None, None):
        merged.time_window = new.time_window

    # Override explicit_constraints (union)
    if new.explicit_constraints:
        merged.explicit_constraints = list(dict.fromkeys(previous.explicit_constraints + new.explicit_constraints))

    # Override wants_flight if new explicitly sets it
    if new.wants_flight:
        merged.wants_flight = True

    # Override confidence
    merged.confidence = new.confidence

    # Override is_revision
    merged.is_revision = True

    # Override is_actionable and actionability_followup from new
    merged.is_actionable = new.is_actionable
    merged.actionability_followup = new.actionability_followup

    # Override revision_op from new if present
    if new.revision_op is not None:
        merged.revision_op = new.revision_op

    # Override confirm_op from new if present
    if new.confirm_op is not None:
        merged.confirm_op = new.confirm_op

    return merged


def _iterative_actionability_check(intent: Intent) -> None:
    """Force is_actionable=False if city or meal_slots are missing after merge."""
    # Skip if there is a pending replacement (already handled by collision detection)
    if intent.pending_replacement is not None:
        return
    if intent.is_actionable:
        missing = []
        if intent.city is None:
            missing.append("city")
        if not intent.meal_slots:
            missing.append("meal_slots")
        if missing:
            intent.is_actionable = False
            # Generate appropriate follow-up
            if "meal_slots" in missing and "city" not in missing:
                intent.actionability_followup = (
                    "好的，已為您記錄。但請問您的用餐時段是\n"
                    "(A) 午餐\n(B) 晚餐\n(C) 宵夜"
                )
            elif "city" in missing and "meal_slots" not in missing:
                intent.actionability_followup = _DEFAULT_MISSING_CITY_FOLLOWUP
            else:
                intent.actionability_followup = (
                    "請指定城市與用餐時段：\n"
                    "(A) 東京 — 午餐\n(B) 東京 — 晚餐\n(C) 大阪 — 午餐\n(D) 大阪 — 晚餐"
                )
            print(f"[State Manager] Forced is_actionable=False due to missing: {missing}")


def _sanitize_intent(intent: Intent) -> None:
    """Remove any fields not in the Intent dataclass (defensive)."""
    # The Intent dataclass already defines allowed fields.
    # We can also ensure meal_slots only contain valid values.
    valid_slots = {"breakfast", "lunch", "tea", "dinner", "late_night"}
    intent.meal_slots = [s for s in intent.meal_slots if s in valid_slots]
    valid_modes = {"right_now", "balanced", "taste_max"}
    if intent.mode not in valid_modes:
        intent.mode = "balanced"
    valid_regions = {"tw", "jp", "unknown"}
    if intent.region not in valid_regions:
        intent.region = "unknown"
    # Ensure city is None if empty string
    intent.city = _strip_city_optional(intent.city)


def _check_global_schedule_collision(intent: Intent, global_schedule: dict | None = None) -> Intent:
    """Detect slot collisions with the global schedule and create pending_replacement if needed.

    If a meal slot is already occupied in the global schedule and the new category_tags
    introduce tags not already present, the intent is marked non-actionable and a
    two‑phase commit (A/B) is set up.
    """
    # If there is already a pending replacement, do not create another.
    if intent.pending_replacement is not None:
        return intent

    if global_schedule is None:
        global_schedule = get_global_schedule()
    for slot in intent.meal_slots:
        if slot not in global_schedule:
            continue
        existing_tags = set(global_schedule[slot])
        new_tags = set(intent.category_tags)
        # If the new tags are already covered by the existing tags, no collision.
        if new_tags.issubset(existing_tags):
            continue

        # Collision detected – block the intent and set up pending replacement.
        print(f"[State Manager] Slot collision: {slot} already has {existing_tags}, new tags {new_tags}")
        intent.is_actionable = False
        intent.pending_replacement = {
            "meal_slots": [slot],
            "category_tags": list(new_tags),
        }
        # Restore category_tags to the existing tags (keep state clean).
        intent.category_tags = list(existing_tags)
        # Generate the A/B follow‑up.
        existing_str = "、".join(existing_tags)
        new_str = "、".join(new_tags)
        intent.actionability_followup = (
            f"【{slot}】時段已排定【{existing_str}】相關行程。"
            f"請問您要將【{slot}】換成【{new_str}】嗎？\n"
            f"(A) 換掉原本【{slot}】\n"
            f"(B) 取消本次操作"
        )
        return intent

    return intent


def _audit_itinerary_for_closed_days(
    prev_itinerary: list[dict],
    new_start_date: str,
    catalog: dict[str, ShopProfile],
) -> list[dict]:
    """Check each shop in prev_itinerary against its closed_days for the new date.

    Returns a list of conflict dicts: {"day": int, "shop_name": str, "weekday": int}.
    """
    from datetime import datetime, timedelta
    conflicts: list[dict] = []
    try:
        start_dt = datetime.strptime(new_start_date, "%Y-%m-%d")
    except (ValueError, TypeError):
        return conflicts
    for entry in prev_itinerary:
        day_offset = entry.get("day", 0)
        shop_name = entry.get("shop_name", "")
        if not shop_name:
            continue
        shop = catalog.get(shop_name)
        if shop is None:
            continue
        closed = getattr(shop, "closed_weekdays", [])
        if not closed:
            continue
        target_date = start_dt + timedelta(days=day_offset)
        weekday = target_date.weekday()
        if weekday in closed:
            conflicts.append({
                "day": day_offset,
                "shop_name": shop_name,
                "weekday": weekday,
            })
    return conflicts


# ---------------------------------------------------------------------------
# Public parse functions
# ---------------------------------------------------------------------------

_LLM_SYSTEM_PROMPT = """\
You are a structured intent extractor for a food-travel planning assistant.
Given a user query, output ONLY a JSON object — no markdown, no explanation.

JSON schema:
{
  "city": "<specific city name in original language, or JSON null if unknown — NEVER invent a city>",
  "region": "<'tw' for Taiwan | 'jp' for Japan | 'unknown'>",
  "meal_slots": ["<breakfast|lunch|tea|dinner|late_night>", ...],
  "time_window": {"start": "<HH:MM or null>", "end": "<HH:MM or null>"},
  "category_tags": ["<food category tags e.g. ramen, sushi, dessert, izakaya>", ...],
  "dietary_hints": "<vegan|vegetarian|pescatarian|no_beef|no_ramen|no_pork|null>",
  "excluded_shops": ["<canonical venue name(s) the user refuses, e.g. full storefront>", ...],
  "excluded_tags": ["<lowercase category/cuisine tags to avoid, e.g. matcha, cafe, coffee>", ...],
  "must_include_shops": ["<shop name(s) user explicitly mandates to keep, e.g. '不要動 XX'>"],
  "must_exclude_shops": ["<shop name(s) user explicitly mandates to drop, e.g. '換掉 XX'>"],
  "mode": "<right_now|balanced|taste_max>",
  "explicit_constraints": ["<appetite_light|strong_ramen|...>"],
  "wants_flight": <true|false>,
  "confidence": <0.0-1.0>,
  "is_revision": <false for a fresh standalone query>,
  "is_actionable": <true|false>,
  "actionability_followup": "<string or null>"
}

ACTIONABILITY (is_actionable + actionability_followup)
------------------------------------------------------
Decide if the request is executable for restaurant/itinerary search WITHOUT guessing user's geography or meal structure.

Set is_actionable=false when ANY of these hold:
- User names only a country/region but no city AND no implicit locale from context (e.g. 「我想去日本」).
- User names food/category but no city/time/meal slot and none can be inferred (e.g. 「我想吃拉麵」 alone).

When is_actionable=false you MUST set actionability_followup to a concise question in the user's language (Traditional Chinese for zh requests).
The follow-up MUST list exactly four reply choices labeled (A) (B) (C) (D) on separate lines or clearly separated.
You MUST write the FULL TEXT of each choice, NOT just letters. For example:
(A) 關東地區（如東京）
(B) 關西地區（如大阪、京都）
(C) 北海道
(D) 其他（請自行輸入）
Never output only letters like "(A) (B) (C) (D)" without explanatory content.
Replace the above example with options appropriate to the user's context (country, region, etc.). Do not hardcode Japanese cities for a non-Japan query.

When is_actionable=true, set actionability_followup to null.

MISSING PARAMETER CLARIFICATION (meal_slots)
--------------------------------------------
If the user's query implies a FULL-DAY trip or itinerary (e.g. "一日行程", "排行程", "整天", "一天", "full day"), set meal_slots=["breakfast","lunch","tea","dinner"] and is_actionable=true. Do NOT ask about meal slots for full-day queries.

Only ask about meal slots when the user clearly wants a SINGLE meal (e.g. "吃拉麵", "找晚餐", "午餐去哪") but has not specified which meal slot. In that case set is_actionable=false and ask with (A)(B)(C)(D) options.

CRITICAL RULE — NO CITY GUESSING (ABSOLUTE)
-------------------------------------------
You MUST NOT guess, invent, or implicitly default any city (including Kyoto/Tokyo/Osaka).
If the user names only a country (e.g. 「日本」「台灣」) or a broad area without naming ONE concrete city
(東京、京都、大阪、台北…), set `"city": null`, set `is_actionable` to **false**, and put region-level (A)(B)(C)(D)
choices in `actionability_followup`.
Only output a non-null `city` when the user (or thread context) clearly names or implies that single city.

CRITICAL RULE — STRICT NEGATION HANDLING (WANTED VS UNWANTED)
-------------------------------------------------------------
You must strictly distinguish between wanted and unwanted entities.
- IF NEGATIVE ("不想", "不要", "不吃", "不喜歡", "避開", "換掉"): you MUST put the item in `excluded_tags` or `dietary_hints`. `category_tags` MUST NOT contain it.
  * Example: "不想吃麵" => excluded_tags: ["ramen", "noodle"], category_tags MUST BE EMPTY or unrelated.
  * Example: "不吃牛" => excluded_tags: ["beef"], dietary_hints: "no_beef", category_tags MUST NOT contain beef.
- IF POSITIVE ("想吃", "要", "喜歡"): you MUST put the item in `category_tags`.
DO NOT confuse these two. Explicitly negated items NEVER go into category_tags.

CRITICAL RULE — SELF-CORRECTION
-------------------------------
If the user's query has a corrective tone, pointing out that you overlooked previously provided information (e.g., "我剛剛就說過吃晚餐了", "上面不是有寫午餐嗎"), you MUST:
- Extract the correct meal slot(s) and place them in meal_slots (e.g., ["dinner"]).
- Still set is_actionable=false (so the system can broadcast an apology message).
- In actionability_followup output a self-correction and apology, for example: "非常抱歉，我漏看了您已經指定了晚餐時段！我立刻為您處理。"

Rules:
- mode=right_now when user is hungry NOW or wants nearby results within 30 min.
- mode=taste_max when food quality is the main focus.
- mode=balanced otherwise.
- excluded_shops: named venues only (specific restaurant/bar names). use [] if none.
- excluded_tags: ONLY for food categories/cuisines/vibes the user does NOT want.
  Use lowercase retrieval tags (e.g. ramen, noodle, matcha, cafe, beef, spicy).
  Never put venue names here. Use [] if none.
- category_tags: ONLY for food categories the user DOES want. Never put negations here.
- dietary_hints: use for ethical/medical/religious restrictions only (vegan, no_beef, no_pork, etc.).
- confidence: 0.0 if city/meal intent is completely unclear; 1.0 if all fields are explicit.
- is_revision: always false here.
- is_actionable / actionability_followup: follow ACTIONABILITY rules above.
- Do NOT add examples or commentary. Output JSON only.

Few-shot examples demonstrating CORRECT negation handling:
Query: '不想吃麵'
Output: {"city":null,"region":"unknown","meal_slots":[],"time_window":{"start":null,"end":null},"category_tags":[],"dietary_hints":"no_ramen","excluded_shops":[],"excluded_tags":["ramen","noodle","udon","soba"],"mode":"balanced","explicit_constraints":[],"wants_flight":false,"confidence":0.6,"is_revision":false,"is_actionable":true,"actionability_followup":null}

Query: '不吃牛'
Output: {"city":null,"region":"unknown","meal_slots":[],"time_window":{"start":null,"end":null},"category_tags":[],"dietary_hints":"no_beef","excluded_shops":[],"excluded_tags":["beef","yakiniku","wagyu"],"mode":"balanced","explicit_constraints":[],"wants_flight":false,"confidence":0.6,"is_revision":false,"is_actionable":true,"actionability_followup":null}

Few-shot — NOT actionable (must include A–D in actionability_followup):
Query: '我想去日本'
Output: {"city":null,"region":"jp","meal_slots":[],"time_window":{"start":null,"end":null},"category_tags":[],"dietary_hints":null,"excluded_shops":[],"excluded_tags":[],"mode":"balanced","explicit_constraints":[],"wants_flight":false,"confidence":0.3,"is_revision":false,"is_actionable":false,"actionability_followup":"日本很大，您較想先規劃哪個區域？\\n(A) 北海道\\n(B) 關東（東京周邊）\\n(C) 關西（大阪／京都）\\n(D) 九州／沖繩"}

Query: '我想吃拉麵'
Output: {"city":null,"region":"unknown","meal_slots":[],"time_window":{"start":null,"end":null},"category_tags":["ramen"],"dietary_hints":null,"excluded_shops":[],"excluded_tags":[],"mode":"taste_max","explicit_constraints":[],"wants_flight":false,"confidence":0.35,"is_revision":false,"is_actionable":false,"actionability_followup":"想在哪個城市找拉麵？\\n(A) 東京\\n(B) 大阪\\n(C) 京都\\n(D) 台北"}

Few-shot — Missing Parameter Clarification (Needs meal slot):
Query: '我想去京都吃燒肉'
Output: {"city":"京都","region":"jp","meal_slots":[],"time_window":{"start":null,"end":null},"category_tags":["yakiniku"],"dietary_hints":null,"excluded_shops":[],"excluded_tags":[],"mode":"balanced","explicit_constraints":[],"wants_flight":false,"confidence":0.8,"is_revision":false,"is_actionable":false,"actionability_followup":"請問您想安排在哪個時段享用呢？\\n(A) 午餐\\n(B) 晚餐\\n(C) 宵夜"}

Few-shot — Self-Correction (User points out missed meal slot):
Query: '我剛剛就說過要吃晚餐了'
Output: {"city":null,"region":"unknown","meal_slots":["dinner"],"time_window":{"start":null,"end":null},"category_tags":[],"dietary_hints":null,"excluded_shops":[],"excluded_tags":[],"mode":"balanced","explicit_constraints":[],"wants_flight":false,"confidence":0.9,"is_revision":false,"is_actionable":false,"actionability_followup":"非常抱歉，我漏看了您已經指定了晚餐時段！我立刻為您處理。"}

Few-shot — Full-day itinerary with implicit meal slots:
Query: '排京都美食行程'
Output: {"city":"京都","region":"jp","meal_slots":["breakfast","lunch","tea","dinner"],"time_window":{"start":null,"end":null},"category_tags":[],"dietary_hints":null,"excluded_shops":[],"excluded_tags":[],"mode":"taste_max","explicit_constraints":[],"wants_flight":false,"confidence":0.95,"is_revision":false,"is_actionable":true,"actionability_followup":null}

Query: '幫我排台北一日行程 不要咖啡廳'
Output: {"city":"台北","region":"tw","meal_slots":["breakfast","lunch","tea","dinner"],"time_window":{"start":null,"end":null},"category_tags":[],"dietary_hints":null,"excluded_shops":[],"excluded_tags":["cafe","coffee"],"mode":"balanced","explicit_constraints":[],"wants_flight":false,"confidence":0.95,"is_revision":false,"is_actionable":true,"actionability_followup":null}

# Revision operations
- "把燃えよ麺助換掉" → revision_op: {op_type: "replace", target_shop: "燃えよ麺助", new_shop: null, slot_id: null}
- "不要第二餐" → revision_op: {op_type: "remove", target_shop: "<第二個 slot 的 shop_name>", new_shop: null, slot_id: null}
- "把燃えよ麺助換成麵屋武士" → revision_op: {op_type: "replace", target_shop: "燃えよ麺助", new_shop: "麵屋武士", slot_id: null}

若非修改行程的 query，revision_op 輸出 null。

# Confirm / save itinerary
- 「這樣滿意了，幫我存檔，京都第一天」→ confirm_op: {message: "京都第一天"}
- 「OK存起來」→ confirm_op: {message: ""}
- 若非存檔請求，confirm_op 輸出 null。
"""

_LLM_REFINEMENT_SYSTEM_PROMPT = """\
You revise structured intent for a food-travel assistant in MULTI-TURN refinement mode.

Workflow
--------
1) Compare `previous_intent` JSON (first user message below) against `New user message` (second message).
2) Emit ONE JSON object describing the authoritative intent AFTER this turn. No markdown.

Schema — same keys as cold extraction plus `is_revision`, `is_actionable`, `actionability_followup`:
{
  "city": "<specific city | JSON null if still unknown — NEVER invent defaults>",
  "region": "'tw' | 'jp' | 'unknown'",
  "meal_slots": ["breakfast"|"lunch"|"tea"|"dinner"|"late_night", ...],
  "time_window": {"start": "<HH:MM or null>", "end": "<HH:MM or null>"},
  "category_tags": ["..."],
  "dietary_hints": "<vegan|vegetarian|pescatarian|no_beef|no_ramen|no_pork|null>",
  "excluded_shops": ["<specific venue(s) user refuses>", ...],
  "excluded_tags": ["<lowercase tags to steer away from, e.g. matcha, cafe>", ...],
  "must_include_shops": ["<shop name(s) user explicitly mandates to keep, e.g. '不要動 XX'>"],
  "must_exclude_shops": ["<shop name(s) user explicitly mandates to drop, e.g. '換掉 XX'>"],
  "mode": "<right_now|balanced|taste_max>",
  "explicit_constraints": ["..."],
  "wants_flight": <true|false>,
  "confidence": <0.0-1.0>,
  "is_revision": <true|false>,
  "is_actionable": <true|false>,
  "actionability_followup": "<string or null; if false, MUST include (A)(B)(C)(D)>"
}

CRITICAL RULE 1 — FIELD INHERITANCE FOR REVISION (DO NOT DROP FIELDS!)
----------------------------------------------------------------------
When the user says they want to swap/replace/exclude a specific venue (e.g. '把XX換掉', '不要XX', 'replace XX', 'swap XX'), this is a venue-level change ONLY.
You MUST carry forward ALL of these fields from previous_intent unchanged:
- city
- region  
- meal_slots (CRITICAL: NEVER reduce the number of meal slots during a venue swap)
- time_window
- mode

Only update excluded_shops (add the venue to exclude) and set is_revision=true.
Example:
previous_intent has meal_slots=["breakfast","lunch","tea","dinner"], city="京都"
user says: "把燃えよ麺助換掉"
correct output: carry forward meal_slots=["breakfast","lunch","tea","dinner"], city="京都", add "燃えよ麺助" to excluded_shops, is_revision=true
WRONG output: meal_slots=["breakfast","lunch"], city="unknown"

CRITICAL RULE 2 — STRICT NEGATION HANDLING (WANTED VS UNWANTED)
---------------------------------------------------------------
You must strictly distinguish between wanted and unwanted entities.
- IF NEGATIVE ("不想", "不要", "不吃"): you MUST put the item in `excluded_tags` or `dietary_hints`. `category_tags` MUST NOT contain it.
  Example: "不想吃麵" => excluded_tags: ["ramen", "noodle"], category_tags MUST BE EMPTY or unrelated.
- IF POSITIVE ("想吃", "要"): you MUST put the item in `category_tags`.
DO NOT confuse these two.

CRITICAL RULE 3 — NO CITY GUESSING (ABSOLUTE)
----------------------------------------------
Same as cold extraction: never invent or default a city. If the user still names only a country/region without ONE concrete city,
keep `"city": null`, set `is_actionable` false, and supply (A)(B)(C)(D) in `actionability_followup`.
You MUST write the FULL TEXT of each choice, NOT just letters. Provide concrete options suitable for the user's context (e.g., city or region names). Never output only letters.

Industry intent-refinement playbook
-----------------------------------
* **Additive / supplement** (“也要有素食”“少油一點”“預算低”): Carry forward geography from `previous_intent`.
* **Venue avoidance** (“不想吃XX”“換掉XX”): append to `excluded_shops`. `is_revision` true.
* **Slot-level food swap** (“把午餐換成蕎麥麵”): Infer targeted `meal_slots`; update `category_tags` while carrying forward the rest. `is_revision` true.

MULTI-TURN SHORT REPLY RESOLUTION (answer to previous actionability_followup)
-------------------------------------------------------------------------------
CRITICAL RULE: If the user's reply is a single letter (like "A", "B", "C", "D") or a short selection, you MUST examine the (A)(B)(C)(D) options that the model previously wrote in `previous_intent.actionability_followup`. Map the user's short answer to one of those options, extract the corresponding city name, and set:
- `"city"` to that city name.
- `"is_actionable"` = true
- `"actionability_followup"` = null

For example:
   - Input "A" or "(A)" or "選A" ⇒ the first option.
   - Input a city name (e.g. "東京") ⇒ the option containing that city.

When the user’s new message is SHORT (e.g. "A", "B", "第一個", "東京", "大阪") and
`previous_intent.is_actionable` is False and `previous_intent.actionability_followup` is set:

1) Examine the (A)(B)(C)(D) options that the model previously wrote in `previous_intent.actionability_followup`.
2) Map the user’s short answer to one of those options.  For example:
   - Input "A" or "(A)" or "選A" ⇒ the first option.
   - Input a city name (e.g. "東京") ⇒ the option containing that city.
3) Once a concrete city is identified, set:
   - `"city"` to that city name (e.g. "東京").
   - `"region"` appropriately ("tw" for Taipei, "jp" for Japanese cities).
   - `"is_actionable"` = true
   - `"actionability_followup"` = null
4) Carry forward any other fields that were already set in `previous_intent` (dietary_hints, category_tags, …).
   The `is_revision` flag should be true.

CRITICAL RULE 4 — MUST INCLUDE / MUST EXCLUDE (HARD LOCK)
-----------------------------------------------------------
When `is_revision` is true, carefully examine the user's new message and identify any shops the
user explicitly wants to keep (e.g. "保留 XX" / "keep XX") and any shops they want to remove
(e.g. "換掉 XX" / "不要 XX").  Fill `must_include_shops` with the shop name(s) they explicitly
mandate to keep, and `must_exclude_shops` with shop name(s) they mandate to drop.
If no such explicit mandates exist, leave both as empty lists.
CRITICAL RULE — SELF-CORRECTION
-------------------------------
If the user's query has a complaining or corrective tone, pointing out that you overlooked previously provided information (e.g., "我上一句就說過我要吃晚餐了啊", "你沒看到嗎"), you MUST:
- Fully retain the city and category_tags from previous_intent (NEVER move wanted food into excluded_tags).
- Extract the correct meal slot(s) and place them in meal_slots (e.g., ["dinner"]).
- Set is_actionable=true (because the conditions are now complete; the system must continue planning). As a special case, even though is_actionable is true, you MUST still output an apology in actionability_followup, for example: "非常抱歉，我漏看了您已經指定了晚餐時段！我立刻為您規劃。"
- In actionability_followup output an apology, for example: "非常抱歉，我漏看了您已經指定了晚餐時段！我立刻為您規劃。"

Few-shot — 「不要動」= 保留（must_include_shops），「不要」= 排除（excluded_shops）:
previous_intent: {"city": "京都", "meal_slots": ["breakfast", "lunch", "tea", "dinner"], ...}
Query: '松籟庵不要動，換晚餐'
Output: {"city":"京都","region":"jp","meal_slots":["breakfast","lunch","tea","dinner"],"time_window":{"start":null,"end":null},"category_tags":[],"dietary_hints":null,"excluded_shops":[],"excluded_tags":[],"must_include_shops":["松籟庵"],"must_exclude_shops":[],"mode":"balanced","explicit_constraints":[],"wants_flight":false,"confidence":0.9,"is_revision":true,"is_actionable":true,"actionability_followup":null,"revision_op":{"op_type":"replace","target_shop":null,"new_shop":null,"slot_id":null},"confirm_op":null,"metadata":{}}
說明：「不要動」= 保留，放進 must_include_shops。excluded_shops 保持空。

Few-shot — 「不要 XX」= 排除（excluded_shops）:
previous_intent: {"city": "京都", "meal_slots": ["breakfast", "lunch", "tea", "dinner"], ...}
Query: '不要拉麵店，換午餐'
Output: {"city":"京都","region":"jp","meal_slots":["breakfast","lunch","tea","dinner"],"time_window":{"start":null,"end":null},"category_tags":[],"dietary_hints":null,"excluded_tags":["ramen","noodle"],"excluded_shops":[],"must_include_shops":[],"must_exclude_shops":[],"mode":"balanced","explicit_constraints":[],"wants_flight":false,"confidence":0.9,"is_revision":true,"is_actionable":true,"actionability_followup":null,"revision_op":null,"confirm_op":null,"metadata":{}}
說明：「不要拉麵店」是食材/類型排除，放進 excluded_tags，不影響 excluded_shops。

Few-shot — Self-Correction during revision:
previous_intent: {"city": "京都", "category_tags": ["yakiniku"], "meal_slots": [], ...}
Query: '我上一句就說過我要吃晚餐了啊，你沒看到嗎'
Output: {"city":"京都","region":"jp","meal_slots":["dinner"],"time_window":{"start":null,"end":null},"category_tags":["yakiniku"],"dietary_hints":null,"excluded_shops":[],"excluded_tags":[],"mode":"balanced","explicit_constraints":[],"wants_flight":false,"confidence":0.9,"is_revision":true,"is_actionable":true,"actionability_followup":"非常抱歉，我漏看了您已經指定了晚餐時段！我立刻為您處理。"}

# Confirm / save itinerary
- 「這樣滿意了，幫我存檔，京都第一天」→ confirm_op: {message: "京都第一天"}
- 「OK存起來」→ confirm_op: {message: ""}
- 若非存檔請求，confirm_op 輸出 null。
"""

def _prev_itinerary_system_addon(prev_itinerary: str | None) -> str:
    """Append prior-round markdown itinerary so the model can interpret amendment-style queries."""
    pw = (prev_itinerary or "").strip()
    if not pw:
        return ""
    max_chars = 6000
    body = pw[:max_chars] + ("…" if len(pw) > max_chars else "")
    return (
        "\n\n---\nCurrent itinerary baseline (this conversation thread). "
        "The user may refer to specific meals or sections here. "
        "Prefer amendment / partial update over discarding the whole plan unless they ask to replan.\n"
        "<<<ITINERARY\n"
        f"{body}\n"
        "ITINERARY>>>\n"
    )


def _llm_prompt_messages(
    query: str,
    previous_intent: Intent | None,
    *,
    prev_itinerary: str | None = None,
) -> list[dict[str, Any]]:
    """Build chat messages for INTENT_PARSING (cold extraction vs refinement)."""
    itinerary_ctx = _prev_itinerary_system_addon(prev_itinerary)
    if previous_intent is None:
        return [
            {"role": "system", "content": _LLM_SYSTEM_PROMPT + itinerary_ctx},
            {"role": "user", "content": f"User query: {query}"},
        ]
    snapshot = dict(previous_intent.as_dict())
    snapshot.pop("is_revision", None)
    snapshot.pop("confidence", None)
    # -- Dynamic context injection for short reply resolution --
    followup = (previous_intent.actionability_followup or "").strip()
    context_line = ""
    if followup:
        context_line = (
            f"\n（系統先前詢問了使用者：{followup}。"
            f"請根據這個選項清單，解析使用者最新的簡短回覆。）\n"
        )
    return [
        {"role": "system", "content": _LLM_REFINEMENT_SYSTEM_PROMPT + itinerary_ctx},
        {"role": "user", "content": json.dumps({"previous_intent": snapshot}, ensure_ascii=False)},
        {"role": "user", "content": f"New user message: {query}{context_line}"},
    ]


def parse_intent_rules(
    query: str,
    *,
    user_locale: str = "zh_TW",
    user_lat: Optional[float] = None,
    user_lng: Optional[float] = None,
    previous_intent: Optional['Intent'] = None,
) -> Optional['Intent']:
    print(f"👉 [DEBUG-RULE] 快慢路徑攔截器收到的原始 query: {query!r}")
    
    q = query.strip().upper()
    
    # Handle pending mutation confirmation (A/B)
    if previous_intent is not None and previous_intent.pending_mutation is not None:
        option_match = re.match(r"^[(（]?\s*([A-B])\s*[)）]?(?:\s|$|.)", q)
        if option_match:
            choice = option_match.group(1)
            base = copy.deepcopy(previous_intent)
            mutation = base.pending_mutation
            if choice == "A":
                # Apply pending mutation
                if "meal_slots" in mutation:
                    base.meal_slots = list(mutation["meal_slots"])
                elif "remove_shops" in mutation:
                    # Remove those shops from the itinerary (mark as excluded)
                    remove_shops = mutation["remove_shops"]
                    base.excluded_shops = list(
                        dict.fromkeys(base.excluded_shops + remove_shops)
                    )
                base.pending_mutation = None
                base.is_actionable = True
                base.actionability_followup = None
                base.confidence = 1.0
                print(f"👉 [DEBUG-RULE] 使用者確認變更，套用 pending_mutation: {mutation}")
                return base
            else:  # choice == "B"
                # Reject pending mutation
                base.pending_mutation = None
                base.is_actionable = True
                base.actionability_followup = None
                base.confidence = 1.0
                print(f"👉 [DEBUG-RULE] 使用者拒絕變更，清除 pending_mutation")
                return base
    
    # Handle pending replacement confirmation (A/B/C/D)
    if previous_intent is not None and previous_intent.pending_replacement is not None:
        option_match = re.match(r"^[(（]?\s*([A-D])\s*[)）]?(?:\s|$|.)", q)
        if option_match:
            choice = option_match.group(1)
            base = copy.deepcopy(previous_intent)
            replacement = base.pending_replacement
            slot = replacement.get("meal_slots", [None])[0]
            added_tags = replacement.get("added_tags", [])
            old_tags = replacement.get("old_tags", [])
            if choice == "A":
                # Apply pending replacement: update global schedule if slot present
                if slot is not None:
                    global_schedule = get_global_schedule()
                    global_schedule[slot] = list(added_tags)
                    set_global_schedule(global_schedule)
                # Apply the new tags to the intent
                base.category_tags = list(added_tags)
                base.pending_replacement = None
                base.is_actionable = True
                base.actionability_followup = None
                base.confidence = 1.0
                print(f"👉 [DEBUG-RULE] 使用者確認換掉，更新 global_schedule slot {slot} -> {added_tags}")
                return base
            elif choice == "B":
                # Reject pending replacement: keep existing schedule, remove the slot from meal_slots if present
                if slot is not None and slot in base.meal_slots:
                    base.meal_slots.remove(slot)
                base.pending_replacement = None
                base.is_actionable = True
                base.actionability_followup = None
                base.confidence = 1.0
                print(f"👉 [DEBUG-RULE] 使用者取消換掉，移除 slot {slot}")
                return base
            elif choice == "C":
                # Both: new tags before old tags
                base.category_tags = list(old_tags) + list(added_tags)
                new_str = "、".join(added_tags)
                old_str = "、".join(old_tags)
                ec = list(base.explicit_constraints)
                ec.append(f"必須將 {new_str} 安排在 {old_str} 之前")
                base.explicit_constraints = ec
                base.pending_replacement = None
                base.is_actionable = True
                base.actionability_followup = None
                base.confidence = 1.0
                print(f"👉 [DEBUG-RULE] 使用者選擇兩個都吃，{new_str} 排在 {old_str} 之前")
                return base
            elif choice == "D":
                # Both: new tags after old tags
                base.category_tags = list(old_tags) + list(added_tags)
                new_str = "、".join(added_tags)
                old_str = "、".join(old_tags)
                ec = list(base.explicit_constraints)
                ec.append(f"必須將 {new_str} 安排在 {old_str} 之後")
                base.explicit_constraints = ec
                base.pending_replacement = None
                base.is_actionable = True
                base.actionability_followup = None
                base.confidence = 1.0
                print(f"👉 [DEBUG-RULE] 使用者選擇兩個都吃，{new_str} 排在 {old_str} 之後")
                return base
    
    option_match = re.match(r"^[(（]?\s*([A-D])\s*[)）]?(?:\s|$|.)", q)
    if option_match:
        choice = option_match.group(1)
        mapping = {"A": "東京", "B": "大阪", "C": "京都", "D": "台北"}
        city = mapping[choice]
        region = "jp" if city != "台北" else "tw"
        print(f"👉 [DEBUG-RULE] 攔截成功！選項 {choice} 映射為 {city}")
        
        if previous_intent is not None:
            base = copy.deepcopy(previous_intent)
            base.city = city
            base.region = region
            base.is_actionable = True
            base.confidence = 1.0
            base.actionability_followup = None
            return base
        return Intent(
            city=city,
            region=region,
            is_actionable=True,
            confidence=1.0,
        )

    known_cities = ["東京", "大阪", "京都", "台北"]
    if q in known_cities:
        region = "jp" if q != "台北" else "tw"
        print(f"👉 [DEBUG-RULE] 攔截成功！關鍵字匹配為 {q}")
        
        if previous_intent is not None:
            base = copy.deepcopy(previous_intent)
            base.city = q
            base.region = region
            base.is_actionable = True
            base.confidence = 1.0
            base.actionability_followup = None
            return base
        return Intent(
            city=q,
            region=region,
            is_actionable=True,
            confidence=1.0,
        )
        
    # Full-day itinerary pattern: 排 X 行程 / 幫我排 / 安排行程
    import re as _re
    _ITINERARY_PATTERNS = [
        r'排.{0,10}行程',
        r'安排.{0,10}行程',
        r'規劃.{0,10}行程',
        r'幫我排.{0,10}行程',
        r'幫我安排.{0,10}行程',
        r'一日遊',
        r'整天行程',
        r'全天行程',
    ]
    if any(_re.search(p, query) for p in _ITINERARY_PATTERNS):
        # Extract city from query if present
        city, region, _ = _resolve_city(query, user_locale, user_lat, user_lng)
        if city:
            result = Intent(
                city=city,
                region=region,
                meal_slots=["breakfast", "lunch", "tea", "dinner"],
                is_actionable=True,
                confidence=0.9,
            )
            # Carry forward previous_intent constraints if available
            if previous_intent is not None:
                result.excluded_tags = list(previous_intent.excluded_tags or [])
                result.dietary_hints = previous_intent.dietary_hints
                result.mode = previous_intent.mode or "balanced"
            return result

    print("👉 [DEBUG-RULE] 攔截失敗，準備進入 LLM...")
    return None

def parse_intent_llm(
    query: str,
    llm_router: Any,
    *,
    previous_intent: Intent | None = None,
    prev_itinerary: str | None = None,
) -> Intent:
    """LLM-based extraction or refinement with pydantic validation and one retry.

    On schema mismatch the failure is recorded in the active OTEL span and
    a single retry is attempted with an additional hint in the prompt.
    Raises ``ValueError`` if both attempts fail validation.
    """
    from llm_router import TaskType  # local import avoids circular deps at module load

    messages_base = _llm_prompt_messages(query, previous_intent, prev_itinerary=prev_itinerary)

    # === 在呼叫 LLM 之前加入這段 ===
    print("\n👉 [DEBUG-PROMPT] 即將送給 LLM 的訊息:")
    import json
    print(json.dumps(messages_base, ensure_ascii=False, indent=2))
    print("=" * 50)
    # ================================
    
    last_error: Exception | None = None
    for attempt in range(2):
        messages = list(messages_base)
        if attempt == 1 and last_error is not None:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Your previous response failed validation: {last_error}. "
                        "Please output ONLY a valid JSON object matching the schema."
                    ),
                }
            )

        tracer = _get_tracer()
        span_name = f"intent_parser.llm_attempt_{attempt}"

        def _do_call(msgs: list[dict]) -> Intent:
            response = llm_router.complete(TaskType.INTENT_PARSING, msgs)
            raw = (response.content or "").strip()
            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = re.sub(r"^```[a-z]*\n?", "", raw)
                raw = re.sub(r"\n?```$", "", raw)
            parsed_json = json.loads(raw)
            validated = _LLMIntentSchema(**parsed_json)
            return _schema_to_intent(validated)

        try:
            if tracer is not None:
                with tracer.start_as_current_span(span_name) as span:
                    span.set_attribute("intent_parser.attempt", attempt)
                    span.set_attribute("intent_parser.query_len", len(query))
                    span.set_attribute("intent_parser.refinement", previous_intent is not None)
                    result = _do_call(messages)
                    span.set_attribute("intent_parser.success", True)
                    return result
            else:
                return _do_call(messages)
        except (json.JSONDecodeError, Exception) as exc:
            last_error = exc
            if tracer is not None:
                with tracer.start_as_current_span(span_name) as span:
                    span.set_attribute("intent_parser.attempt", attempt)
                    span.set_attribute("intent_parser.error", str(exc)[:200])
                    span.set_attribute("intent_parser.success", False)

    raise ValueError(
        f"parse_intent_llm: both attempts failed for query {query!r}. "
        f"Last error: {last_error}"
    )


def parse_intent(
    query: str,
    llm_router: Any,
    *,
    user_locale: str | None = None,
    user_lat: float | None = None,
    user_lng: float | None = None,
    previous_intent: Intent | None = None,
    prev_itinerary: str | None = None,
    global_schedule: dict | None = None,
    catalog: dict[str, ShopProfile] | None = None,
    new_start_date: str | None = None,
    itinerary_slots: list[dict] | None = None,
) -> Intent:
    """Hybrid parser: refinement LLM path when ``previous_intent`` is supplied; else rules → LLM.

    This entry point never raises LLM failures: it falls back to rules or partial intents.

    Geographic pin: when ``user_lat``/``user_lng`` fall inside a known bounded box (e.g. Taipei),
    ``city``/``region`` are set from coordinates — never from a global default like Kyoto.
    """

    def _stamp_geo_pins(intent_obj: Intent) -> None:
        geo = _coords_to_city_region(user_lat, user_lng)
        if geo is not None:
            intent_obj.city = geo[0]
            intent_obj.region = geo[1]

    # --- Closed‑day audit when date changes ---
    if previous_intent is not None and new_start_date is not None and catalog is not None:
        prev_date = previous_intent.metadata.get("trip_start_date")
        if prev_date is not None and prev_date != new_start_date:
            import json
            try:
                itinerary_list = json.loads(prev_itinerary) if prev_itinerary else []
            except (json.JSONDecodeError, TypeError):
                itinerary_list = []
            if itinerary_list:
                conflicts = _audit_itinerary_for_closed_days(
                    itinerary_list, new_start_date, catalog
                )
                if conflicts:
                    lines = [f"已幫您將出發日更改為 {new_start_date}。但注意："]
                    for c in conflicts:
                        day_num = c["day"] + 1
                        lines.append(
                            f"原本排在 Day {day_num} 的【{c['shop_name']}】剛好遇到公休！"
                        )
                    lines.append("")
                    lines.append("請問要幫您把這家店取消換成別的，還是要維持原日期呢？")
                    lines.append("(A) 幫我取消公休的店並重排")
                    lines.append("(B) 算了，維持原本的日期")
                    followup = "\n".join(lines)
                    remove_shops = [c["shop_name"] for c in conflicts]
                    pending = {"remove_shops": remove_shops}
                    result = copy.deepcopy(previous_intent)
                    result.is_actionable = False
                    result.actionability_followup = followup
                    result.pending_mutation = pending
                    result.metadata["trip_start_date"] = new_start_date
                    _sanitize_intent(result)
                    return result

    # First, try rule-based fast path regardless of previous_intent
    rule_result = parse_intent_rules(
        query, user_locale=user_locale, user_lat=user_lat, user_lng=user_lng,
        previous_intent=previous_intent,
    )
    if rule_result is not None and rule_result.confidence >= 0.6:
        _stamp_geo_pins(rule_result)
        _clamp_missing_city_if_actionable(rule_result)
        return rule_result

    if previous_intent is not None:
        try:
            llm_result = parse_intent_llm(
                query,
                llm_router,
                previous_intent=previous_intent,
                prev_itinerary=prev_itinerary,
            )
            _stamp_geo_pins(llm_result)
            llm_result = _reconcile_intents(
                previous_intent, llm_result,
                global_schedule=global_schedule,
                query=query,
                llm_router=llm_router,
                itinerary_slots=itinerary_slots,
            )
            # Rule-based shop lock post-processor
            if llm_result.is_revision:
                _extract_shop_lock_ops(query, itinerary_slots, llm_result)
            _iterative_actionability_check(llm_result)
            _sanitize_intent(llm_result)
            _clamp_missing_city_if_actionable(llm_result)
            return llm_result
        except Exception:
            return parse_intent(
                query,
                llm_router,
                user_locale=user_locale,
                user_lat=user_lat,
                user_lng=user_lng,
                previous_intent=None,
                prev_itinerary=prev_itinerary,
                global_schedule=global_schedule,
                catalog=catalog,
                new_start_date=new_start_date,
            )

    try:
        llm_result = parse_intent_llm(query, llm_router, prev_itinerary=prev_itinerary)
        _stamp_geo_pins(llm_result)
        llm_result = _check_global_schedule_collision(llm_result, global_schedule=global_schedule)
        _iterative_actionability_check(llm_result)
        _sanitize_intent(llm_result)
        _clamp_missing_city_if_actionable(llm_result)
        return llm_result
    except Exception:
        if rule_result is not None:
            _sanitize_intent(rule_result)
            return rule_result
        city, region, _ = _resolve_city(query, user_locale, user_lat, user_lng)
        fallback = Intent(
            city=city,
            region=region,
            is_actionable=False,
            actionability_followup=(
                "無法從這則訊息解析出可執行的行程需求，請選擇或補充：\n"
                "(A) 日本 — 關西\n(B) 日本 — 關東\n(C) 台灣\n(D) 其他（請直接輸入城市）"
            ),
        )
        _sanitize_intent(fallback)
        return fallback
