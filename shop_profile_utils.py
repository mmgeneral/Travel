from __future__ import annotations
import re
import math
import logging
from datetime import datetime, timedelta
from typing import Any

from shop_planning import ShopProfile, BookingType, QueueStrategy, FlavorCategory, AuthorityData

logger = logging.getLogger(__name__)


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


def _make_cache_key(meal_type: str, intent_dict: dict) -> str:
    tags = frozenset(
        (intent_dict.get("excluded_tags") or []) +
        (intent_dict.get("category_tags") or []) +
        ([intent_dict["dietary_hints"]] if intent_dict.get("dietary_hints") else [])
    )
    return str((meal_type, tuple(sorted(tags))))


def _schedule_slots(slots: list[dict], shop_catalog: dict) -> list[dict]:
    """
    Assign start_time to each slot sequentially.
    Default start: 09:00, duration: 90 min per slot.
    Travel time between slots: use haversine distance estimate
    (2 min per km, minimum 5 min).
    Now also checks feasibility (cooldown, open time) and records warnings.
    """
    from feasibility_utils import can_transition, estimate_travel_minutes, SLOT_CLOCK_HOUR_MINUTE
    import math
    DEFAULT_START = "09:00"
    DEFAULT_DURATION = 90  # minutes

    def _get_shop_obj(name: str):
        obj = shop_catalog.get(name)
        if isinstance(obj, dict):
            return _dict_to_shop(obj) if "name" in obj else None
        return obj  # Assume it's already a ShopProfile

    def _dict_to_shop(d: dict) -> object:
        from shop_planning import ShopProfile, BookingType, QueueStrategy, FlavorCategory, AuthorityData
        # minimal conversion to get coordinates and basic fields needed for feasibility
        return type("_", (), {
            "name": d.get("name",""),
            "open_time": d.get("open_time",""),
            "close_time": d.get("close_time",""),
            "base_wait_minutes": int(d.get("base_wait_minutes",0)),
            "min_eat_minutes": int(d.get("min_eat_minutes",d.get("avg_eat_minutes",60))),
            "avg_eat_minutes": int(d.get("avg_eat_minutes",60)),
            "latitude": float(d.get("latitude",0.0)),
            "longitude": float(d.get("longitude",0.0)),
            "flavor_category": FlavorCategory.LIGHT,
            "has_small_portion": bool(d.get("has_small_portion",False)),
        })()  # type: ignore

    current_time = datetime.strptime(DEFAULT_START, "%H:%M")
    previous_start_dt: datetime | None = None
    previous_shop_obj = None

    for i, slot in enumerate(slots):
        cur_shop_name = slot.get("shop_name")
        cur_shop_obj = _get_shop_obj(cur_shop_name)

        # ----- locked slot (already has a fixed start_time) -----
        if slot.get("user_locked") and slot.get("start_time"):
            try:
                current_time = datetime.strptime(slot["start_time"], "%H:%M")
                current_time += timedelta(minutes=slot.get("duration_minutes") or DEFAULT_DURATION)
            except ValueError:
                pass
            cur_start_dt = datetime.strptime(slot["start_time"], "%H:%M")

            # check transition from previous slot to this locked slot
            if i > 0 and previous_start_dt is not None and previous_shop_obj is not None and cur_shop_obj is not None:
                feasible, reason = can_transition(
                    previous_start_dt, previous_shop_obj,
                    cur_start_dt, cur_shop_obj,
                    mode="BALANCED",
                    requested_meal_count=None,
                    appetite_light_mode=False,
                )
                if not feasible:
                    slot["feasibility_warning"] = reason

            previous_start_dt = cur_start_dt
            previous_shop_obj = cur_shop_obj
            continue   # no further travel addition to maintain original behaviour

        # ----- unlocked slot: assign start_time, duration -----
        meal_type = slot.get("meal_type", "")
        if meal_type in SLOT_CLOCK_HOUR_MINUTE:
            hh, mm = SLOT_CLOCK_HOUR_MINUTE[meal_type]
            anchor_dt = current_time.replace(hour=hh, minute=mm, second=0, microsecond=0)
            if current_time < anchor_dt:
                current_time = anchor_dt
        slot["start_time"] = current_time.strftime("%H:%M")
        slot["duration_minutes"] = slot.get("duration_minutes") or DEFAULT_DURATION
        cur_start_dt = datetime.strptime(slot["start_time"], "%H:%M")
        # advance clock after finishing this meal
        current_time += timedelta(minutes=slot["duration_minutes"])

        # check transition from previous slot to this unlocked slot
        if i > 0 and previous_start_dt is not None and previous_shop_obj is not None and cur_shop_obj is not None:
            feasible, reason = can_transition(
                previous_start_dt, previous_shop_obj,
                cur_start_dt, cur_shop_obj,
                mode="BALANCED",
                requested_meal_count=None,
                appetite_light_mode=False,
            )
            if not feasible:
                slot["feasibility_warning"] = reason

        previous_start_dt = cur_start_dt
        previous_shop_obj = cur_shop_obj

        # add travel time to the next slot (skipped for locked slots and for the very last slot)
        if i + 1 < len(slots):
            next_shop_name = slots[i + 1].get("shop_name", "")
            if cur_shop_name and next_shop_name and cur_shop_obj is not None:
                next_shop_obj = _get_shop_obj(next_shop_name)
                if next_shop_obj is not None:
                    travel_min = estimate_travel_minutes(cur_shop_obj, next_shop_obj)
                    current_time += timedelta(minutes=travel_min)
                else:
                    current_time += timedelta(minutes=15)
            else:
                current_time += timedelta(minutes=15)

    return slots
