"""
Hybrid intent parser: rule fast-path → LLM fallback.

Public API
----------
parse_intent(query, llm_router, *, user_locale, user_lat, user_lng) -> Intent
parse_intent_rules(query, *, user_locale, user_lat, user_lng)        -> Intent | None
parse_intent_llm(query, llm_router)                                  -> Intent

Intent dataclass fields
-----------------------
city               str          e.g. "台北" / "京都" / "東京"
region             str          "tw" | "jp" | "unknown"
meal_slots         list[str]    subset of breakfast/lunch/tea/dinner/late_night
time_window        tuple[str|None, str|None]  (HH:MM start, HH:MM end) or Nones
category_tags      list[str]    e.g. ["ramen", "dessert"]
dietary_hints      str | None   "vegan" | "vegetarian" | "pescatarian" | None
mode               str          "right_now" | "balanced" | "taste_max"
explicit_constraints list[str]  e.g. ["appetite_light", "strong_ramen"]
wants_flight       bool
confidence         float        0.0–1.0 (rules estimate)

Design notes
------------
* Rule helpers are private (_-prefixed) and live here; agent.py no longer
  imports them directly.
* LLM output is validated with pydantic; a single retry is attempted on
  schema mismatch, with OTEL span attributes recorded.
* parse_intent() skips the LLM entirely when rules return confidence >= 0.6.
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
    mode: str = "balanced"
    explicit_constraints: list[str] = field(default_factory=list)
    wants_flight: bool = False
    confidence: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation for AgentState storage."""
        return {
            "city": self.city,
            "region": self.region,
            "meal_slots": list(self.meal_slots),
            "time_window": list(self.time_window),
            "category_tags": list(self.category_tags),
            "dietary_hints": self.dietary_hints,
            "mode": self.mode,
            "explicit_constraints": list(self.explicit_constraints),
            "wants_flight": self.wants_flight,
            "confidence": self.confidence,
        }


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
    q = (query or "").lower()
    if "vegan" in q or "純素" in q:
        return "vegan"
    if "vegetarian" in q or "素食" in q:
        return "vegetarian"
    if "pescatarian" in q:
        return "pescatarian"
    return None


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
    mode: str = "balanced"
    explicit_constraints: list[str] = []
    wants_flight: bool = False
    confidence: float = 0.8

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
        mode=str(s.mode),
        explicit_constraints=list(s.explicit_constraints),
        wants_flight=bool(s.wants_flight),
        confidence=float(s.confidence),
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
  "dietary_hints": "<vegan|vegetarian|pescatarian|null>",
  "mode": "<right_now|balanced|taste_max>",
  "explicit_constraints": ["<appetite_light|strong_ramen|...>"],
  "wants_flight": <true|false>,
  "confidence": <0.0-1.0>
}

Rules:
- mode=right_now when user is hungry NOW or wants nearby results within 30 min.
- mode=taste_max when food quality is the main focus.
- mode=balanced otherwise.
- If city is unclear, leave it blank ("") and set region="unknown".
- confidence reflects how sure you are of the extracted intent (0=not sure, 1=very sure).
- Do NOT add examples or commentary. Output JSON only.
"""


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
        mode=mode,
        explicit_constraints=explicit_constraints,
        wants_flight=wants_flight,
        confidence=confidence,
    )


def parse_intent_llm(query: str, llm_router: Any) -> Intent:
    """LLM-based intent extraction with pydantic validation and one retry.

    On schema mismatch the failure is recorded in the active OTEL span and
    a single retry is attempted with an additional hint in the prompt.
    Raises ``ValueError`` if both attempts fail validation.
    """
    from llm_router import TaskType  # local import avoids circular deps at module load

    messages_base = [
        {"role": "system", "content": _LLM_SYSTEM_PROMPT},
        {"role": "user", "content": f"User query: {query}"},
    ]

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
) -> Intent:
    """Hybrid parser: rules first, LLM fallback when confidence < 0.6.

    This is the primary entry point.  It never raises; on LLM failure it
    returns a best-effort rule-derived Intent with confidence=0.0.
    """
    rule_result = parse_intent_rules(
        query, user_locale=user_locale, user_lat=user_lat, user_lng=user_lng
    )

    if rule_result is not None and rule_result.confidence >= 0.6:
        return rule_result

    # Fallback to LLM
    try:
        return parse_intent_llm(query, llm_router)
    except Exception:
        # LLM failed entirely; return the rule result if we have one, else default
        if rule_result is not None:
            return rule_result
        city, region, _ = _resolve_city(query, user_locale, user_lat, user_lng)
        return Intent(city=city, region=region)
