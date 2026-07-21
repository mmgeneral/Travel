from __future__ import annotations
import copy
import re
import json
import logging
from typing import Any, TYPE_CHECKING

from debug_json import debug_json as _dj
from llm_router import TaskType
from decision_engine import ItinerarySynthesizer

if TYPE_CHECKING:
    from agent import AgentState

logger = logging.getLogger(__name__)

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


def _consume_pending_dietary_clarification_answer(state: AgentState, llm_router: Any) -> bool:
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


def _llm_extract_city_from_choice(
    llm_router: Any,
    question: str,
    user_answer: str,
    state: AgentState,
) -> str | None:
    """Minimal LLM call: map user answer to a city name from the given clarification options."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a city extractor. Given the previous options list and the user's answer, "
                "output JUST the city name (e.g. 東京, 台北). No extra text."
            ),
        },
        {
            "role": "user",
            "content": f"Options:\n{question}\n\nUser answer: {user_answer}",
        },
    ]
    try:
        response = llm_router.complete(TaskType.INTENT_PARSING, messages)
        raw = (response.content or "").strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
        raw = raw.strip()
        if raw:
            return raw
    except Exception:
        pass
    return None


def _apply_city_from_clarification(state: AgentState, city: str) -> None:
    """Fill `state['intent']` with the resolved city and mark actionable."""
    region = "jp" if city in ("東京", "京都", "大阪") else "tw" if city == "台北" else "unknown"
    state["intent"] = {
        "city": city,
        "region": region,
        "meal_slots": [],
        "time_window": [None, None],
        "category_tags": [],
        "dietary_hints": None,
        "excluded_shops": [],
        "excluded_tags": [],
        "mode": "balanced",
        "explicit_constraints": [],
        "wants_flight": False,
        "confidence": 0.9,
        "is_revision": True,
        "is_actionable": True,
        "actionability_followup": None,
    }
    state.pop("awaiting_intent_clarification", None)
    state.pop("clarification_broadcast", None)
    state.setdefault("transit_audit", []).append(
        _dj("intent_clarification_resolved", city=city, method="interceptor")
    )


def _consume_pending_intent_clarification(state: AgentState, llm_router: Any) -> bool:
    """
    Slot‑filling interceptor for intent clarification (A/B/C/D options).
    Called at the start of node_route_intent *before* any LLM re‑parse.

    Returns True if the clarification answer was successfully resolved,
    causing node_route_intent to early‑return with the filled intent.
    """
    if not state.get("awaiting_intent_clarification"):
        return False

    broadcast = state.get("clarification_broadcast") or {}
    question = str(broadcast.get("question", "") or "")
    q = (state.get("query") or "").strip()
    if not q:
        # nothing to parse, clear the flag anyway so we don’t loop forever
        state.pop("awaiting_intent_clarification", None)
        state.pop("clarification_broadcast", None)
        return False

    # 1) Try a lightweight heuristic mapping for most common answer patterns
    #    (single letter, parenthesised letter, or a known city name)
    q_lower = q.lower().strip()
    known_cities = {"東京": "jp", "东京": "jp", "京都": "jp", "大阪": "jp",
                    "台北": "tw", "taipei": "tw", "tokyo": "jp",
                    "kyoto": "jp", "osaka": "jp"}
    direct_match = None
    for k, reg in known_cities.items():
        if k in q_lower or q_lower in k:
            direct_match = k
            break
    if direct_match:
        _apply_city_from_clarification(state, direct_match)
        return True

    # 2) Answer is a short letter code – rely on LLM to map it
    city = _llm_extract_city_from_choice(llm_router, question, q, state)
    if city:
        _apply_city_from_clarification(state, city)
        return True

    # 3) fallback: clear the flag and let the original LLM parsing run
    state.pop("awaiting_intent_clarification", None)
    state.pop("clarification_broadcast", None)
    return False
