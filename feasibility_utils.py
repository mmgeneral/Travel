from __future__ import annotations
import math
import logging
from datetime import datetime, timedelta
from typing import Optional
from shop_planning import ShopProfile, FlavorCategory

logger = logging.getLogger(__name__)

SLOT_CLOCK_HOUR_MINUTE: dict[str, tuple[int, int]] = {
    "breakfast": (8, 0),
    "lunch": (12, 0),
    "tea": (15, 0),
    "dinner": (18, 30),
    "late_night": (21, 30),
}


def calculate_cooldown(
    prev_shop: ShopProfile | None,
    *,
    mode: str = "BALANCED",
    requested_meal_count: int | None = None,
    appetite_light_mode: bool = False,
) -> int:
    from decision_engine import OptimizationMode

    _mode = OptimizationMode(mode) if isinstance(mode, str) else mode
    if prev_shop is None:
        return 90
    cat = prev_shop.flavor_category
    minutes: int
    if cat == FlavorCategory.HEAVY:
        if appetite_light_mode and requested_meal_count is not None and requested_meal_count >= 3:
            minutes = 60
        elif _mode == OptimizationMode.TASTE_MAX and requested_meal_count is not None:
            minutes = 75
        else:
            minutes = 90
    elif cat == FlavorCategory.SWEET:
        minutes = 30
    elif cat == FlavorCategory.LIGHT:
        minutes = 60
    elif cat == FlavorCategory.REFRESHING:
        minutes = 45
    else:
        minutes = 60
    if bool(getattr(prev_shop, "has_small_portion", False)):
        minutes = max(25, int(round(float(minutes) * 0.88)))
    return minutes


def shop_open_at(base: datetime, shop: ShopProfile) -> datetime:
    raw = getattr(shop, "open_time", None) or "11:00"
    parts = str(raw).strip().split(":", 1)
    hh = int(parts[0])
    mm = int(parts[1]) if len(parts) > 1 else 0
    return base.replace(hour=hh, minute=mm, second=0, microsecond=0)


def shop_open_close_window(base: datetime, shop: ShopProfile) -> tuple[datetime, datetime]:
    open_at = shop_open_at(base, shop)
    close_h, close_m = [int(x) for x in shop.close_time.split(":", 1)]
    close_at = base.replace(hour=close_h, minute=close_m, second=0, microsecond=0)
    if close_at <= open_at:
        close_at = close_at + timedelta(days=1)
    return open_at, close_at


def estimate_travel_minutes(shop_a: ShopProfile, shop_b: ShopProfile, *, fallback_minutes: int = 15) -> int:
    km = _haversine_km(shop_a.latitude, shop_a.longitude, shop_b.latitude, shop_b.longitude)
    minutes = max(5, int(km * 2))  # 2 min per km
    return minutes


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def can_transition(
    a_start: datetime,
    a_shop: ShopProfile | None,
    b_start: datetime,
    b_shop: ShopProfile | None,
    *,
    mode: str = "BALANCED",
    requested_meal_count: int | None = None,
    appetite_light_mode: bool = False,
) -> tuple[bool, str]:
    """
    Returns (feasible, reason). If feasible, reason is an empty string.
    Checks cooldown, travel time, and whether the target shop is open.
    If either shop is None, always returns True with empty reason.
    """
    if a_shop is None or b_shop is None:
        return True, ""
    from decision_engine import OptimizationMode
    _mode = OptimizationMode(mode) if isinstance(mode, str) else mode

    travel_min = estimate_travel_minutes(a_shop, b_shop)
    cooldown = calculate_cooldown(a_shop, mode=_mode, requested_meal_count=requested_meal_count, appetite_light_mode=appetite_light_mode)
    fastest_finish_at = a_start + timedelta(
        minutes=int(a_shop.base_wait_minutes) + int(a_shop.min_eat_minutes or a_shop.avg_eat_minutes)
    )
    ready_at = fastest_finish_at + timedelta(minutes=cooldown) + timedelta(minutes=travel_min)
    open_b = shop_open_at(b_start, b_shop)

    if ready_at <= b_start and b_start >= open_b:
        return True, ""

    reasons: list[str] = []
    if ready_at > b_start:
        reasons.append(
            f"ready after start by {(ready_at - b_start).seconds // 60} min "
            f"(cooldown={cooldown} travel={travel_min})"
        )
    if b_start < open_b:
        reasons.append(f"shop not yet open (open at {open_b.strftime('%H:%M')})")
    return False, "; ".join(reasons) if reasons else "unknown"
