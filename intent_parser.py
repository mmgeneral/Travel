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
city               str          e.g. "台北" / "京都" / "東京"
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

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, field_validator, model_validator

from observability import record_llm_call, _get_tracer

# ---------------------------------------------------------------------------
# Intent data structure
# ---------------------------------------------------------------------------

@dataclass
class Intent:
    city: str = "京都"
    region: str = "jp"
    meal_slots: list[str] = field(default_factory=list)
    time_window: tuple[str | None, str | None] = (None, None)
    category_tags: list[str] = field(default_factory=list)
    dietary_hints: str | None = None
    excluded_shops: list[str] = field(default_factory=list)
    excluded_tags: list[str] = field(default_factory=list)
    mode: str = "balanced"
    explicit_constraints: list[str] = field(default_factory=list)
    wants_flight: bool = False
    confidence: float = 0.0
    #: True when this intent updates a stored prior snapshot (refinement turn).
    is_revision: bool = False

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation for AgentState storage."""
        return {
            "city": self.city,
            "region": self.region,
            "meal_slots": list(self.meal_slots),
            "time_window": list(self.time_window),
            "category_tags": list(self.category_tags),
            "dietary_hints": self.dietary_hints,
            "excluded_shops": list(self.excluded_shops),
            "excluded_tags": list(self.excluded_tags),
            "mode": self.mode,
            "explicit_constraints": list(self.explicit_constraints),
            "wants_flight": self.wants_flight,
            "confidence": self.confidence,
            "is_revision": self.is_revision,
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
    return Intent(
        city=str(d.get("city") or "京都"),
        region=str(d.get("region") or "jp"),
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
        mode=str(d.get("mode") or "balanced"),
        explicit_constraints=[str(x) for x in (d.get("explicit_constraints") or []) if x is not None],
        wants_flight=bool(d.get("wants_flight", False)),
        confidence=float(d.get("confidence", 0.0)),
        is_revision=bool(d.get("is_revision", False)),
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

    return ("京都", "jp", False)


# ---------------------------------------------------------------------------
# Pydantic schema for LLM output validation
# ---------------------------------------------------------------------------

_VALID_SLOTS = {"breakfast", "lunch", "tea", "dinner", "late_night"}
_VALID_MODES = {"right_now", "balanced", "taste_max"}
_VALID_REGIONS = {"tw", "jp", "unknown"}


class _LLMIntentSchema(BaseModel):
    city: str = "京都"
    region: str = "jp"
    meal_slots: list[str] = []
    time_window: dict[str, str | None] = {"start": None, "end": None}
    category_tags: list[str] = []
    dietary_hints: str | None = None
    excluded_shops: list[str] = []
    excluded_tags: list[str] = []
    mode: str = "balanced"
    explicit_constraints: list[str] = []
    wants_flight: bool = False
    confidence: float = 0.8
    is_revision: bool = False

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


def _schema_to_intent(s: _LLMIntentSchema) -> Intent:
    tw = s.time_window or {}
    return Intent(
        city=str(s.city or "京都"),
        region=str(s.region or "jp"),
        meal_slots=list(s.meal_slots),
        time_window=(tw.get("start"), tw.get("end")),
        category_tags=list(s.category_tags),
        dietary_hints=s.dietary_hints,
        excluded_shops=list(s.excluded_shops),
        excluded_tags=list(s.excluded_tags),
        mode=str(s.mode),
        explicit_constraints=list(s.explicit_constraints),
        wants_flight=bool(s.wants_flight),
        confidence=float(s.confidence),
        is_revision=bool(s.is_revision),
    )


# ---------------------------------------------------------------------------
# Public parse functions
# ---------------------------------------------------------------------------

_LLM_SYSTEM_PROMPT = """\
You are a structured intent extractor for a food-travel planning assistant.
Given a user query, output ONLY a JSON object — no markdown, no explanation.

JSON schema:
{
  "city": "<city name in original language, e.g. 台北/東京/京都>",
  "region": "<'tw' for Taiwan | 'jp' for Japan | 'unknown'>",
  "meal_slots": ["<breakfast|lunch|tea|dinner|late_night>", ...],
  "time_window": {"start": "<HH:MM or null>", "end": "<HH:MM or null>"},
  "category_tags": ["<food category tags e.g. ramen, sushi, dessert, izakaya>", ...],
  "dietary_hints": "<vegan|vegetarian|pescatarian|no_beef|no_ramen|no_pork|null>",
  "excluded_shops": ["<canonical venue name(s) the user refuses, e.g. full storefront>", ...],
  "excluded_tags": ["<lowercase category/cuisine tags to avoid, e.g. matcha, cafe, coffee>", ...],
  "mode": "<right_now|balanced|taste_max>",
  "explicit_constraints": ["<appetite_light|strong_ramen|...>"],
  "wants_flight": <true|false>,
  "confidence": <0.0-1.0>,
  "is_revision": <false for a fresh standalone query>
}

Rules:
- mode=right_now when user is hungry NOW or wants nearby results within 30 min.
- mode=taste_max when food quality is the main focus.
- mode=balanced otherwise.
- excluded_shops: named venues only (specific restaurant/bar names). use [] if none.
- excluded_tags: ONLY for food categories/cuisines/vibes the user does NOT want.
  Use lowercase retrieval tags (e.g. ramen, noodle, matcha, cafe, beef, spicy).
  Never put venue names here. Use [] if none.
- category_tags: ONLY for food categories the user DOES want. Never put negations here.
- CRITICAL RULE — negation handling:
  If the user says they do NOT want something (不想、不要、不吃、避開、換掉、no X、avoid X),
  put the tag in excluded_tags. NEVER put it in category_tags.
  If the user says they DO want something, put the tag in category_tags.
- dietary_hints: use for ethical/medical/religious restrictions only (vegan, no_beef, no_pork, etc.).
- confidence: 0.0 if city/meal intent is completely unclear; 1.0 if all fields are explicit.
- is_revision: always false here.
- Do NOT add examples or commentary. Output JSON only.

Few-shot examples (these show correct mapping — study them carefully):

Query: '我要去京都吃拉麵'
Output: {'city':'京都','region':'jp','category_tags':['ramen'],'excluded_tags':[],'dietary_hints':null,'confidence':0.9}

Query: '不想吃麵'
Output: {'city':'','region':'unknown','category_tags':[],'excluded_tags':['ramen','noodle','udon','soba','tsukemen'],'dietary_hints':null,'confidence':0.5}

Query: '幫我排行程，不要拉麵'
Output: {'city':'','region':'unknown','category_tags':[],'excluded_tags':['ramen'],'dietary_hints':'no_ramen','confidence':0.6}

Query: '我不吃牛肉，幫我排京都行程'
Output: {'city':'京都','region':'jp','category_tags':[],'excluded_tags':['beef','yakiniku','wagyu'],'dietary_hints':'no_beef','confidence':0.85}

Query: '不要抹茶的店'
Output: {'city':'','region':'unknown','category_tags':[],'excluded_tags':['matcha'],'dietary_hints':null,'confidence':0.6}

Query: '我想吃壽司，不要太貴'
Output: {'city':'','region':'unknown','category_tags':['sushi'],'excluded_tags':['fine_dining','kaiseki'],'dietary_hints':null,'confidence':0.7}

Query: '京都美食之旅，我要吃最好吃的'
Output: {'city':'京都','region':'jp','category_tags':[],'excluded_tags':[],'dietary_hints':null,'mode':'taste_max','confidence':0.8}
"""

_LLM_REFINEMENT_SYSTEM_PROMPT = """\
You revise structured intent for a food-travel assistant in MULTI-TURN refinement mode.

Workflow
--------
1) Compare `previous_intent` JSON (first user message below) against `New user message` (second message).
2) Emit ONE JSON object describing the authoritative intent AFTER this turn. No markdown.

Schema — same keys as cold extraction plus `is_revision`:
{
  "city": "...",
  "region": "'tw' | 'jp' | 'unknown'",
  "meal_slots": ["breakfast"|"lunch"|"tea"|"dinner"|"late_night", ...],
  "time_window": {"start": "<HH:MM or null>", "end": "<HH:MM or null>"},
  "category_tags": ["..."],
  "dietary_hints": "<vegan|vegetarian|pescatarian|no_beef|no_ramen|no_pork|null>",
  "excluded_shops": ["<specific venue(s) user refuses>", ...],
  "excluded_tags": ["<lowercase tags to steer away from, e.g. matcha, cafe>", ...],
  "mode": "<right_now|balanced|taste_max>",
  "explicit_constraints": ["..."],
  "wants_flight": <true|false>,
  "confidence": <0.0-1.0>,
  "is_revision": <true|false>
}

Industry intent-refinement playbook
-----------------------------------
* **Additive / supplement** (“也要有素食”“少油一點”“預算低”): Carry forward geography
  (`city`, `region`), `meal_slots`, and `time_window` from `previous_intent` unless explicitly changed.
  Update `category_tags`, `dietary_hints`, `explicit_constraints`, etc. Merge lists without dropping
  unstated facets. Set `is_revision` true whenever the operative plan materially changes vs prior.
* **Destructive pivot** (“改去台北”“算了換東京”“剛說拉麵改壽司”“不要拉麵了”): Replace every conflicting
  field; drop contradictory cuisine tags/meal slots/time constraints; rebuild mode if urgency changes.
  `is_revision` must be true.
* **Venue avoidance** (“不想吃茶寮都路里”“避雷 ○○ 本店”“don’t suggest X”)：append to `excluded_shops`
  using the fullest identifiable venue label (prior turn’s exclusions stay unless clearly overridden).
  `is_revision` true when exclusions change materially.
* Refinement phrase → field examples:
  • “不要抹茶的店” ⇒ `"excluded_tags": ["matcha"]`.
  • '不想吃麵' ⇒ excluded_tags: ['ramen', 'noodle', 'udon', 'soba']
  • “把咖啡廳都換掉” ⇒ `"excluded_tags": ["cafe", "coffee"]`.
  • “不想吃茶寮都路里” ⇒ `"excluded_shops": ["茶寮 都路里 祇園本店"]` — **venue** ⇒ `excluded_shops`, not `excluded_tags`.
  • '不想吃麵' / '不吃麵食' ⇒ excluded_tags: ['ramen','noodle','udon','soba','tsukemen'], dietary_hints: 'no_ramen'
  • '不想吃辣' ⇒ excluded_tags: ['spicy']
  • '想吃拉麵' ⇒ category_tags: ['ramen']  ← positive only, NOT excluded_tags
  • '不要牛肉' ⇒ excluded_tags: ['beef','yakiniku','wagyu'], dietary_hints: 'no_beef'
  CRITICAL: 否定句（不想、不要、不吃）→ excluded_tags。肯定句（想吃、要）→ category_tags。絕不混淆。
* **Neutral ack / same ask**: keep prior semantics; confidence may stay high and `is_revision` false only
  when nothing operative changes.
* **Slot-level food swap** (“把午餐換成蕎麥麵”“晚餐改壽司”): Infer targeted `meal_slots`; update
  `category_tags` / `explicit_constraints` for the substitution while carrying forward the rest of
  `previous_intent`. When a **Current itinerary baseline** block appears in the system message, use it to
  anchor which meal or section the user refers to. `is_revision` must be true when the plan changes.
* Keep `region` consistent with `city` (`台北/台灣`⇒tw, `京都|東京|大阪`⇒jp).
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
    return [
        {"role": "system", "content": _LLM_REFINEMENT_SYSTEM_PROMPT + itinerary_ctx},
        {"role": "user", "content": json.dumps({"previous_intent": snapshot}, ensure_ascii=False)},
        {"role": "user", "content": f"New user message: {query}"},
    ]


def parse_intent_rules(
    query: str,
    *,
    user_locale: str | None = None,
    user_lat: float | None = None,
    user_lng: float | None = None,
) -> Intent | None:
    """Rule-based fast path.

    Returns ``None`` if no meaningful signals are found in the query (i.e.
    the query is too ambiguous for rules and needs LLM interpretation).
    Returns an ``Intent`` with a ``confidence`` score otherwise.
    """
    city, region, city_in_query = _resolve_city(query, user_locale, user_lat, user_lng)
    tr = _extract_time_range(query)
    time_window = (tr.start, tr.end)
    meal_slots = _requested_meal_slots(query)
    category_tags = sorted(_extract_category_tags(query))
    dietary_hints = _dietary_ethics(query)
    excluded_shops_quick = _extract_excluded_shops_quick(query)
    wants_flight = _is_flight_intent(query)
    appetite_light = _is_appetite_light(query)
    ramen = _is_ramen(query)

    # Determine mode
    if _is_right_now_mode(query):
        mode = "right_now"
    elif category_tags or ramen:
        mode = "taste_max"
    else:
        mode = "balanced"

    explicit_constraints: list[str] = []
    if appetite_light:
        explicit_constraints.append("appetite_light")
    if ramen and _requested_meal_count(query) is not None:
        explicit_constraints.append("strong_ramen")

    # Confidence: how much structure did rules extract?
    score = 0.0
    if city_in_query:
        score += 0.20
    elif user_locale or user_lat is not None:
        score += 0.08  # city from context, not query
    if meal_slots:
        score += 0.30
    if category_tags:
        score += 0.20
    if time_window[0] or time_window[1]:
        score += 0.15
    if dietary_hints:
        score += 0.10
    if excluded_shops_quick:
        score += 0.08
    if wants_flight:
        score += 0.10
    if mode == "right_now":
        score += 0.10
    confidence = min(1.0, score)

    # Boost for combinations where rules unambiguously capture the full intent.
    # These avoid unnecessary LLM calls when the query is structurally clear.
    if wants_flight:
        # Flight booking intent is definitively parseable by rules.
        confidence = max(confidence, 0.70)
    if mode == "right_now" and (category_tags or meal_slots):
        # "現在餓了 + 想吃X" is fully handled by rules.
        confidence = max(confidence, 0.70)
    if city_in_query and meal_slots:
        # Explicit city + at least one meal slot = high planning clarity.
        confidence = max(confidence, 0.70)
    if meal_slots and category_tags:
        # Meal structure + food category = enough for deterministic planning.
        confidence = max(confidence, 0.70)

    # No useful signals → caller should use LLM
    has_signal = bool(
        meal_slots
        or category_tags
        or dietary_hints
        or excluded_shops_quick
        or wants_flight
        or mode == "right_now"
        or time_window[0]
    )
    if not has_signal:
        return None

    return Intent(
        city=city,
        region=region,
        meal_slots=meal_slots,
        time_window=time_window,
        category_tags=category_tags,
        dietary_hints=dietary_hints,
        excluded_shops=excluded_shops_quick,
        mode=mode,
        explicit_constraints=explicit_constraints,
        wants_flight=wants_flight,
        confidence=confidence,
    )


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
) -> Intent:
    """Hybrid parser: refinement LLM path when ``previous_intent`` is supplied; else rules → LLM.

    This entry point never raises LLM failures: it falls back to rules or geography defaults.

    Geographic fix: LLM guesses for ``city``/``region`` are overwritten when
    ``user_lat``/``user_lng`` fall inside a known bounded box (e.g. Taipei).
    """

    def _stamp_geo_pins(intent_obj: Intent) -> None:
        geo = _coords_to_city_region(user_lat, user_lng)
        if geo is not None:
            intent_obj.city = geo[0]
            intent_obj.region = geo[1]

    if previous_intent is not None:
        try:
            llm_result = parse_intent_llm(
                query,
                llm_router,
                previous_intent=previous_intent,
                prev_itinerary=prev_itinerary,
            )
            _stamp_geo_pins(llm_result)
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
            )

    rule_result = parse_intent_rules(
        query, user_locale=user_locale, user_lat=user_lat, user_lng=user_lng
    )

    if rule_result is not None and rule_result.confidence >= 0.6:
        return rule_result

    try:
        llm_result = parse_intent_llm(query, llm_router, prev_itinerary=prev_itinerary)
        _stamp_geo_pins(llm_result)
        return llm_result
    except Exception:
        if rule_result is not None:
            return rule_result
        city, region, _ = _resolve_city(query, user_locale, user_lat, user_lng)
        return Intent(city=city, region=region)
