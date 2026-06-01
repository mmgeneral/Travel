from __future__ import annotations
import re
import logging
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from shop_planning import ShopProfile
from decision_engine import WeightProfile, RankedShop

logger = logging.getLogger(__name__)

# TimeRange dataclass moved from agent.py
@dataclass(frozen=True)
class TimeRange:
    start: str | None = None  # HH:MM
    end: str | None = None  # HH:MM


# ── Helper: clamp HH:MM (moved from agent.py) ──────────────────────────────
def _clamp_hhmm_token(hhmm: str) -> str | None:
    try:
        h, mm = [int(x) for x in hhmm.strip().split(":")]
        h = max(0, min(23, h))
        mm = max(0, min(59, mm))
        return f"{h:02d}:{mm:02d}"
    except Exception:
        return None


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

    # e.g. "7點開始", "早上7点开始"
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
    # Strong intent: ramen keyword + explicit meal-planning context
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
    tags = _extract_explicit_category_tags(query)
    direct = _direct_food_category_mentions(query)
    slot_req = _slot_level_required_tags(query)
    out = set(tags)
    if "ramen" not in direct and "ramen" in out:
        if direct & {"dessert", "izakaya"} or slot_req:
            out.discard("ramen")
    return out


def _plan_global_explicit_tags(query: str) -> set[str]:
    softened = _softened_explicit_category_tags(query)
    slot_req = _slot_level_required_tags(query)
    if not slot_req:
        return softened
    slot_union: set[str] = set()
    for v in slot_req.values():
        slot_union |= v
    direct = _direct_food_category_mentions(query)
    if softened and softened <= slot_union and "ramen" not in direct:
        return set()
    return {t for t in softened if t not in slot_union or t in direct}


def _should_damp_preference_for_query(query: str) -> bool:
    if _slot_level_required_tags(query):
        return True
    direct = _direct_food_category_mentions(query)
    return bool(direct & {"dessert", "izakaya"})


def _apply_runtime_weight_damping(wp: WeightProfile) -> WeightProfile:
    return WeightProfile(
        trust_bias=min(1.0, wp.trust_bias + 0.12),
        preference_bias=max(0.05, wp.preference_bias * 0.52),
        logistics_bias=wp.logistics_bias,
    )


# ── Slot-level required tags (uses _requested_meal_slots from agent.py) ──
def _slot_level_required_tags(query: str) -> dict[str, set[str]]:
    # Lazy import to avoid circular dependency
    from agent import _requested_meal_slots as _rm_slots
    slots_plan = _rm_slots(query)
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


# ── Keyword extraction ─────────────────────────────────────────────────────
def _extract_search_category_keywords(query: str) -> list[str]:
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


# ── Seed shop helpers ──────────────────────────────────────────────────────
def _seed_shop_tag_bag(shop: ShopProfile) -> set[str]:
    return {str(t).lower() for t in shop.tags} | {str(t).lower() for t in shop.occasion_tags}


def _seed_single_covers_slot(shop: ShopProfile, slot: str) -> bool:
    needed = _SLOT_TAG_COVERAGE.get(slot)
    if not needed:
        return True
    return bool(_seed_shop_tag_bag(shop) & needed)


def _seed_covers_meal_slot(seed_shops: list[ShopProfile], slot: str) -> bool:
    return any(_seed_single_covers_slot(s, slot) for s in seed_shops)


# ── SLOT_TAG_COVERAGE & forced queries (unchanged) ────────────────────────
_SLOT_TAG_COVERAGE: dict[str, frozenset[str]] = {
    "breakfast": frozenset({"breakfast", "brunch", "morning", "ramen", "cafe"}),
    "lunch": frozenset({"lunch", "main_meal", "ramen", "quick_meal"}),
    "tea": frozenset({"tea", "dessert", "cafe", "cake", "bakery", "afternoon_tea", "coffee", "refresh"}),
    "dinner": frozenset({"dinner", "main_meal", "ramen", "course", "izakaya", "kaiseki", "sukiyaki"}),
    "late_night": frozenset({"late_night", "izakaya", "ramen", "nightlife", "night_food", "late_open"}),
}

_SLOT_FORCED_PLACES_QUERY: dict[str, str] = {
    "breakfast": "breakfast brunch coffee morning cafe egg sandwich",
    "lunch": "lunch restaurant bento noodles quick meal",
    "tea": "dessert shop cake bakery patisserie afternoon tea cafe coffee sweets confectionery",
    "dinner": "izakaya yakitori skewer kushiyaki japanese pub grill dinner beer",
    "late_night": "izakaya ramen late night bar yakitori night snack",
}


# ── Researcher helpers (unchanged) ─────────────────────────────────────────
def _researcher_semantic_overlap(shop: ShopProfile, slot: str) -> bool:
    base = frozenset(_SLOT_TAG_COVERAGE.get(slot, frozenset()))
    if slot == "tea":
        base |= frozenset({"snack", "tangyuan", "tang_yuan"})
    elif slot == "late_night":
        base |= frozenset({"snack", "night_snack"})
    if not base:
        return True
    return bool(_seed_shop_tag_bag(shop) & base)


def _researcher_clock_minutes(hhmm: str | None) -> int | None:
    try:
        parts = str(hhmm or "").strip().split(":", 1)
        hh = max(0, min(23, int(parts[0])))
        mm = max(0, min(59, int(parts[1]) if len(parts) > 1 else 0))
        return hh * 60 + mm
    except Exception:
        return None


def _researcher_slot_required_gate(shop: ShopProfile, slot: str, slot_req: dict[str, set[str]]) -> bool:
    if not slot_req:
        return True
    need = slot_req.get(str(slot).lower())
    if not need:
        return True
    return bool(_seed_shop_tag_bag(shop) & need)


def _researcher_time_fit_slot(shop: ShopProfile, slot: str, tier: str) -> bool:
    if tier == "semantic_only":
        return True
    open_m = _researcher_clock_minutes(getattr(shop, "open_time", "") or "")
    close_m = _researcher_clock_minutes(getattr(shop, "close_time", "") or "")
    bag = _seed_shop_tag_bag(shop)

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
    # Lazy import to avoid circular
    from agent import _requested_meal_slots as _rm_slots
    slots = _rm_slots(query)
    uncovered = [sl for sl in slots if not _seed_covers_meal_slot(seed_shops, sl)]
    intent_kw = _extract_search_category_keywords(query)

    planned: list[str] = []

    for sl in slots:
        bundle = _SLOT_FORCED_PLACES_QUERY.get(sl)
        if bundle:
            planned.append(f"{bundle} {city}")

    base = (query or "").strip()
    if base:
        planned.append(base)

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
