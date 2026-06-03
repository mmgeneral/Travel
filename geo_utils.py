from __future__ import annotations
import os
import re
import math
import logging
from typing import Any

from query_utils import _requested_meal_slots

logger = logging.getLogger(__name__)


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


def _fallback_broad_geo_queries(query: str, city: str, region: str) -> list[str]:
    """Deterministic wider-area queries when LLM is unavailable or fails."""
    # Lazy imports to avoid circular dependency
    from query_utils import _extract_search_category_keywords, _SLOT_FORCED_PLACES_QUERY

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
