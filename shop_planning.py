from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import math
import os
import re
import time
from typing import Any
import requests


class BookingType(str, Enum):
    NONE = "NONE"
    PHONE = "PHONE"
    WEB = "WEB"
    OMAKASE = "OMAKASE"


class QueueStrategy(str, Enum):
    PHYSICAL_LINE = "PHYSICAL_LINE"
    SIGN_UP_SHEET = "SIGN_UP_SHEET"
    TICKET_SYSTEM = "TICKET_SYSTEM"


class FlavorCategory(str, Enum):
    HEAVY = "HEAVY"
    LIGHT = "LIGHT"
    REFRESHING = "REFRESHING"
    SWEET = "SWEET"


class ProbeOutcome(str, Enum):
    OK = "OK"
    FORCE_ABORT = "FORCE_ABORT"


@dataclass
class DietaryAxis:
    ethics: str = "omnivore"  # vegan / vegetarian / pescatarian / omnivore
    allergens: set[str] = field(default_factory=set)
    religious: str = "none"  # halal / kosher / hindu_veg / none
    medical: set[str] = field(default_factory=set)


_ETHICS_ALIASES: dict[str, str] = {
    "unspecified": "unspecified",
    "not_set": "unspecified",
    "vegan": "vegan",
    "vegetarian": "vegetarian",
    "pescatarian": "pescatarian",
    "fish_only": "pescatarian",
    "omnivore": "omnivore",
    "regular": "omnivore",
    "meat": "omnivore",
}


def _normalize_ethics(raw: str) -> str:
    return _ETHICS_ALIASES.get(raw.strip().lower(), "unspecified")


def compute_marketing_noise_score(shop: "ShopProfile") -> float:
    """
    Fame / hype noise prior for UNDERDOG scoring (0 = trustworthy signal, 1 = very noisy).
    Rules:
    - Bib Gourmand or major-chain positioning → high noise (0.8).
    - No awards but stable Google / Tabelog-style scores → low noise (0.1).
    """
    blob = " ".join(
        [
            shop.name.lower(),
            " ".join(str(t).lower() for t in shop.tags),
            " ".join(str(t).lower() for t in shop.occasion_tags),
        ]
    )
    if "畢比登" in blob or "bib gourmand" in blob or "bib_gourmand" in blob or "ビブグルマン" in blob:
        return 0.8
    if "大型連鎖" in blob or "major_chain" in blob:
        return 0.8
    for t in shop.tags:
        ts = str(t).lower()
        if "連鎖" in str(t) or ts in {"chain", "chain_store"}:
            return 0.8

    auth = shop.authority_data
    medal = auth.tablelog_medal or ""
    has_award = bool(
        auth.michelin_star > 0
        or ("百名店" in medal)
        or ("金" in medal)
        or ("銀" in medal)
        or ("銅" in medal)
    )
    tl = max(float(shop.source_scores.get("tablelog", 0.0)), float(shop.source_scores.get("google", 0.0)))
    google_stable = float(shop.google_rating or 0.0) >= 4.0
    stable_ratings = tl >= 0.62 or google_stable

    if not has_award and stable_ratings:
        return 0.1

    return 0.45


@dataclass
class ShopProfile:
    name: str
    #: 0=Monday, 6=Sunday
    closed_weekdays: list[int] = field(default_factory=list)
    close_time: str  # HH:MM
    booking_type: BookingType
    queue_strategy: QueueStrategy
    last_call_offset: int  # minutes before close
    is_cash_only: bool
    sns_handle: str
    avg_eat_minutes: int = 50
    # Lower bound for aggressive schedule compression.
    # If omitted, defaults to ~40% of avg_eat_minutes.
    min_eat_minutes: int | None = None
    is_famous: bool = False
    base_wait_minutes: int = 20
    flavor_intensity: float = 0.5
    flavor_category: FlavorCategory = FlavorCategory.LIGHT
    # Richer flavor representation for foodie-grade similarity/recovery decisions.
    # Keys are intentionally open-ended to support cuisine-specific signals.
    flavor_vector: dict[str, float] = field(default_factory=dict)
    min_lead_hours: int = 0
    base_health_impact: float = 0.5
    customization_score: float = 0.5
    trust_score: float = 0.7
    source_scores: dict[str, float] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    occasion_tags: set[str] = field(default_factory=set)
    neighborhood: str = ""
    authority_data: AuthorityData = field(default_factory=lambda: AuthorityData())
    is_vegan: bool = False
    # For shops that require pre-declared course/menu constraints.
    requires_menu_reservation: bool = False
    allowed_dietary_preferences: list[str] = field(default_factory=list)
    supported_ethics: set[str] = field(default_factory=lambda: {"omnivore", "pescatarian", "vegetarian", "vegan"})
    blocked_allergens: set[str] = field(default_factory=set)
    supported_religious: set[str] = field(default_factory=lambda: {"none"})
    blocked_medical_conditions: set[str] = field(default_factory=set)
    requires_compatibility_check: bool = False
    backup_options: list[str] = field(default_factory=list)
    booking_phone: str = ""
    reservation_channels: list[str] = field(default_factory=list)
    nearby_atm_options: list[str] = field(default_factory=list)
    region: str = "jp"
    closed_weekdays: set[int] = field(default_factory=set)  # 0=Mon .. 6=Sun
    latitude: float | None = None
    longitude: float | None = None
    google_rating: float = 0.0
    open_now: bool | None = None
    opening_hours_today: str = ""
    open_time: str = "11:00"  # HH:MM
    # Portion / sharing semantics for appetite-light or multi-meal scheduling.
    has_small_portion: bool = False  # e.g. half portion, 小盛
    is_sharing_friendly: bool = False  # izakaya-style shared plates
    # 1.0 = strict one-person course; 0.0 = very flexible portions.
    portion_strictness: float = 0.5
    # Hype vs signal (for UNDERDOG_MODE): derived by default from tags + ratings + awards.
    marketing_noise_score: float | None = None

    def __post_init__(self) -> None:
        if self.min_eat_minutes is None:
            self.min_eat_minutes = max(1, int(round(float(self.avg_eat_minutes) * 0.4)))
        else:
            self.min_eat_minutes = max(1, int(self.min_eat_minutes))
        self.portion_strictness = max(0.0, min(1.0, float(self.portion_strictness)))
        if self.marketing_noise_score is None:
            self.marketing_noise_score = compute_marketing_noise_score(self)
        self.marketing_noise_score = max(0.0, min(1.0, float(self.marketing_noise_score)))

    def metadata_verification(
        self,
        dietary_preference: str | DietaryAxis,
        current_time: datetime | None = None,
        arrival_time: datetime | None = None,
    ) -> tuple[bool, str, str]:
        """
        Validate metadata compatibility before scheduling:
        - If the shop requires reserved menu/courses, user's dietary preference
          must be explicitly supported.
        """
        if isinstance(dietary_preference, DietaryAxis):
            axis = dietary_preference
        else:
            pref = dietary_preference.strip().lower()
            axis = DietaryAxis(ethics=pref or "unspecified")

        ethics = _normalize_ethics(axis.ethics)
        allergens = {x.strip().lower() for x in axis.allergens}
        religious = axis.religious.strip().lower()
        medical = {x.strip().lower() for x in axis.medical}
        lead_time_warning = False

        # Time constraint is evaluated independently from dietary semantics.
        if self.min_lead_hours > 0 and current_time is not None and arrival_time is not None:
            lead_hours = (arrival_time - current_time).total_seconds() / 3600.0
            if lead_hours < float(self.min_lead_hours):
                lead_time_warning = True

        # Dedicated preorder expiration risk:
        # lead-time risk is NOT treated as dietary mismatch.
        if lead_time_warning and self.requires_menu_reservation:
            return False, "PREORDER_EXPIRED_RISK", "PREORDER_RISK"

        # Explicit "unspecified" means user did not opt into dietary constraints.
        if ethics == "unspecified":
            return (True, "LEAD_TIME_WARNING", "SUCCESS") if lead_time_warning else (True, "METADATA_OK", "SUCCESS")

        # Fast path: no semantic dietary constraints to verify.
        if ethics == "omnivore" and not allergens and religious in {"", "none"} and not medical:
            return (True, "LEAD_TIME_WARNING", "SUCCESS") if lead_time_warning else (True, "METADATA_OK", "SUCCESS")

        if self.is_vegan and ethics == "omnivore":
            return False, "PREORDER_RISK", "PREORDER_RISK"
        if allergens.intersection({x.lower() for x in self.blocked_allergens}):
            return False, "DIETARY_ALLERGEN_CONFLICT", "FORCE_ABORT"
        if religious not in {"", "none"} and religious not in {x.lower() for x in self.supported_religious}:
            return False, "DIETARY_RELIGIOUS_CONFLICT", "FORCE_ABORT"
        if medical.intersection({x.lower() for x in self.blocked_medical_conditions}):
            return False, "DIETARY_MEDICAL_CONFLICT", "PREORDER_RISK"
        if self.requires_compatibility_check or self.booking_type == BookingType.OMAKASE:
            if ethics not in {x.lower() for x in self.supported_ethics}:
                return False, "COMPATIBILITY_CHECK_REJECTED", "PREORDER_RISK"
        if not self.requires_menu_reservation:
            return (True, "LEAD_TIME_WARNING", "SUCCESS") if lead_time_warning else (True, "METADATA_OK", "SUCCESS")
        if not ethics:
            return False, "METADATA_DIETARY_REQUIRED", "PREORDER_RISK"
        allowed = {p.lower() for p in self.allowed_dietary_preferences}
        if not allowed or ethics in allowed:
            return (True, "LEAD_TIME_WARNING", "SUCCESS") if lead_time_warning else (True, "METADATA_OK", "SUCCESS")
        return False, "METADATA_DIETARY_MISMATCH", "PREORDER_RISK"


@dataclass
class LiveProbeResult:
    outcome: ProbeOutcome
    semantic_status: str
    matched_keyword: str = ""


@dataclass
class PlannedSlot:
    start_at: datetime
    end_at: datetime
    title: str


@dataclass
class ShopPlanResult:
    shop_name: str
    slots: list[PlannedSlot]
    preparation_note: str
    outcome: str
    semantic_status: str
    backup_option: str = ""
    explanation: Explanation | None = None


@dataclass
class Explanation:
    user_facing: str
    operator_trace: str
    expert_signals: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrafficAlert:
    has_alert: bool
    semantic_status: str
    transport_buffer_minutes: int
    suggested_transport: str = "RAIL"
    matched_keyword: str = ""


@dataclass
class DrivingStatus:
    congestion_level: float  # 0.0..1.0
    parking_turnover_risk: float  # 0.0..1.0
    traffic_jitter_minutes: int
    semantic_status: str


@dataclass
class ReviewAnalysisResult:
    final_reliability_score: float
    bias_penalty: float
    filtered_reviews: list[str]
    negative_evidence: list[str]


@dataclass
class AuthorityData:
    tablelog_medal: str = ""
    michelin_star: int = 0
    chef_lineage: list[str] = field(default_factory=list)
    specialty_items: list[str] = field(default_factory=list)
    google_reviews: list[str] = field(default_factory=list)
    review_count: int = 0


class SnsAdapter(ABC):
    @abstractmethod
    def check_store_status(self, sns_handle: str) -> str:
        raise NotImplementedError


class MockSnsProvider(SnsAdapter):
    def __init__(self, fixtures: dict[str, str] | None = None):
        self.fixtures = fixtures or {}

    def check_store_status(self, sns_handle: str) -> str:
        return self.fixtures.get(sns_handle, "")


class SearchApiProvider(SnsAdapter):
    """
    Placeholder for real crawler/search API integration.
    """
    def check_store_status(self, sns_handle: str) -> str:
        _ = sns_handle
        return ""


class NearbySearchTool:
    """
    Dynamic place retrieval adapter.
    Priority:
    1) Google Places API (if GOOGLE_PLACES_API_KEY provided)
    2) OpenStreetMap Nominatim fallback (simulated Google fields)
    """

    #: Text Search returns up to ~20 results per page; 3 pages => up to 60 candidates.
    _MIN_TEXTSEARCH_PAGES = 3
    _NEAR_RADIUS_START_M = 1000
    _NEAR_RADIUS_EXPAND_M = 3000

    def __init__(self, api_key: str | None = None, timeout_s: int = 8):
        self.api_key = api_key or os.getenv("GOOGLE_PLACES_API_KEY", "")
        self.timeout_s = timeout_s
        self._session = requests.Session()

    def search_places(
        self,
        city: str,
        user_query: str,
        limit: int = 60,
        *,
        must_have_tags: list[str] | None = None,
        location_lat: float | None = None,
        location_lng: float | None = None,
    ) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 60))
        if self.api_key:
            results = self._search_google_places(
                city=city,
                user_query=user_query,
                limit=limit,
                must_have_tags=must_have_tags,
                location_lat=location_lat,
                location_lng=location_lng,
            )
            if results:
                return results[:limit]
        return self._search_nominatim(city=city, user_query=user_query, limit=limit)

    @staticmethod
    def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        r = 6371000.0
        p1 = math.radians(lat1)
        p2 = math.radians(lat2)
        dp = math.radians(lat2 - lat1)
        dl = math.radians(lon2 - lon1)
        a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
        return 2 * r * math.asin(math.sqrt(min(1.0, a)))

    def _geocode_city_centroid(self, city: str) -> tuple[float, float] | None:
        q = (city or "").strip()
        if not q:
            return None
        try:
            resp = self._session.get(
                "https://nominatim.openstreetmap.org/search",
                params={"q": q, "format": "jsonv2", "limit": 1},
                headers={"User-Agent": "travel-agent-nearby-search/1.1"},
                timeout=self.timeout_s,
            )
            rows = resp.json() if resp.status_code < 400 else []
        except Exception:
            rows = []
        if not rows:
            return None
        try:
            return float(rows[0]["lat"]), float(rows[0]["lon"])
        except (KeyError, TypeError, ValueError):
            return None

    @staticmethod
    def _raw_place_matches_any_must_have(place: dict[str, Any], must_have: set[str]) -> bool:
        """OR semantics with planner HARD_TAG: place satisfies if it matches any required tag."""
        if not must_have:
            return True
        name_l = (place.get("name") or "").lower()
        types_l = [str(x).lower() for x in place.get("types", [])]
        addr = str(place.get("formatted_address", "") or "").lower()
        blob = f"{name_l} {' '.join(types_l)} {addr}"
        for t in must_have:
            tl = t.lower().strip()
            if not tl:
                continue
            if tl == "ramen":
                if any(k in name_l for k in ("ramen", "ラーメン", "拉麵")) or "ramen" in blob:
                    return True
            elif tl == "dessert":
                if any(k in types_l for k in ("bakery", "cafe", "meal_takeaway", "store")) or any(
                    k in blob for k in ("cake", "dessert", "bakery", "patisserie", "sweet", "confectionery")
                ):
                    return True
            elif tl == "izakaya":
                if any(k in blob for k in ("izakaya", "yakitori", "skewer", "燒鳥", "串燒", "pub", "bar")):
                    return True
            else:
                if tl in blob or tl in types_l:
                    return True
        return False

    def _google_textsearch_pages(
        self,
        *,
        query: str,
        centroid: tuple[float, float] | None,
        radius_m: int | None,
        min_pages: int,
        max_results: int,
    ) -> list[dict[str, Any]]:
        """Fetch Google Places Text Search using next_page_token (≥min_pages when tokens exist)."""
        places: list[dict[str, Any]] = []
        next_page_token = ""
        pages_fetched = 0
        max_pages = max(min_pages, 6)
        invalid_retries = 0

        while pages_fetched < max_pages:
            if pages_fetched > 0:
                if not next_page_token:
                    break
                time.sleep(2.0)

            params: dict[str, Any] = {"key": self.api_key, "language": "zh-TW"}
            if next_page_token:
                params["pagetoken"] = next_page_token
            else:
                params["query"] = query
                if centroid is not None and radius_m is not None and radius_m > 0:
                    lat, lng = centroid
                    params["location"] = f"{lat},{lng}"
                    params["radius"] = int(radius_m)

            try:
                resp = self._session.get(
                    "https://maps.googleapis.com/maps/api/place/textsearch/json",
                    params=params,
                    timeout=self.timeout_s,
                )
                data = resp.json()
            except Exception:
                break

            status = str(data.get("status", "") or "")
            if status == "INVALID_REQUEST" and next_page_token:
                invalid_retries += 1
                if invalid_retries >= 5:
                    break
                time.sleep(2.5)
                continue
            invalid_retries = 0

            if status not in ("OK", "ZERO_RESULTS"):
                break

            batch = data.get("results", []) or []
            if not batch:
                break

            places.extend(batch)
            pages_fetched += 1
            next_page_token = str(data.get("next_page_token", "") or "").strip()

            if len(places) >= max_results:
                break
            if not next_page_token:
                break

        return places[:max_results]

    def _merge_by_place_id(self, *batches: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[str] = set()
        out: list[dict[str, Any]] = []
        for batch in batches:
            for p in batch:
                pid = str(p.get("place_id", "") or "").strip()
                key = pid if pid else f"noid:{(p.get('name') or '')}_{p.get('geometry', {})}"
                if key in seen:
                    continue
                seen.add(key)
                out.append(p)
        return out

    def _search_google_places(
        self,
        city: str,
        user_query: str,
        limit: int,
        must_have_tags: list[str] | None = None,
        location_lat: float | None = None,
        location_lng: float | None = None,
    ) -> list[dict[str, Any]]:
        query = f"{user_query} restaurant in {city}".strip()
        must_set = {t.strip().lower() for t in (must_have_tags or []) if str(t).strip()}

        centroid: tuple[float, float] | None = None
        if location_lat is not None and location_lng is not None:
            try:
                centroid = (float(location_lat), float(location_lng))
            except (TypeError, ValueError):
                centroid = None
        if centroid is None:
            centroid = self._geocode_city_centroid(city)

        raw_batches: list[list[dict[str, Any]]] = []

        if centroid is not None:
            if must_set:
                first = self._google_textsearch_pages(
                    query=query,
                    centroid=centroid,
                    radius_m=self._NEAR_RADIUS_START_M,
                    min_pages=self._MIN_TEXTSEARCH_PAGES,
                    max_results=limit,
                )
                raw_batches.append(first)
                in_1km: list[dict[str, Any]] = []
                for p in first:
                    loc = (p.get("geometry") or {}).get("location") or {}
                    try:
                        plat = float(loc.get("lat", 0.0) or 0.0)
                        plng = float(loc.get("lng", 0.0) or 0.0)
                    except (TypeError, ValueError):
                        continue
                    if self._haversine_m(centroid[0], centroid[1], plat, plng) <= float(self._NEAR_RADIUS_START_M):
                        in_1km.append(p)
                need_expand = not any(self._raw_place_matches_any_must_have(p, must_set) for p in in_1km)
                if need_expand:
                    second = self._google_textsearch_pages(
                        query=query,
                        centroid=centroid,
                        radius_m=self._NEAR_RADIUS_EXPAND_M,
                        min_pages=self._MIN_TEXTSEARCH_PAGES,
                        max_results=limit,
                    )
                    raw_batches.append(second)
            else:
                wide = self._google_textsearch_pages(
                    query=query,
                    centroid=centroid,
                    radius_m=self._NEAR_RADIUS_EXPAND_M,
                    min_pages=self._MIN_TEXTSEARCH_PAGES,
                    max_results=limit,
                )
                raw_batches.append(wide)
        else:
            # No centroid: plain text search (global bias), still paginate to min pages.
            plain = self._google_textsearch_pages(
                query=query,
                centroid=None,
                radius_m=None,
                min_pages=self._MIN_TEXTSEARCH_PAGES,
                max_results=limit,
            )
            raw_batches.append(plain)

        places = self._merge_by_place_id(*raw_batches) if raw_batches else []
        places = places[: max(1, limit)]
        normalized: list[dict[str, Any]] = []
        for p in places:
            loc = (p.get("geometry") or {}).get("location") or {}
            opening = p.get("opening_hours") or {}
            place_id = p.get("place_id", "")
            opening_hours_today = ""
            review_snippets: list[str] = []
            if place_id:
                details = self._fetch_place_details(place_id)
                opening_hours_today = details.get("opening_hours_today", "")
                review_snippets = [str(x) for x in details.get("reviews", [])]
            normalized.append(
                {
                    "name": p.get("name", "").strip(),
                    "lat": float(loc.get("lat", 0.0) or 0.0),
                    "lng": float(loc.get("lng", 0.0) or 0.0),
                    "rating": float(p.get("rating", 0.0) or 0.0),
                    "open_now": opening.get("open_now", None),
                    "opening_hours_today": opening_hours_today,
                    "formatted_address": p.get("formatted_address", ""),
                    "types": [str(x).lower() for x in p.get("types", [])],
                    "reviews": review_snippets,
                }
            )
        return [x for x in normalized if x.get("name")]

    def _fetch_place_details(self, place_id: str) -> dict[str, Any]:
        try:
            resp = self._session.get(
                "https://maps.googleapis.com/maps/api/place/details/json",
                params={
                    "place_id": place_id,
                    "key": self.api_key,
                    "fields": "opening_hours,reviews",
                    "language": "zh-TW",
                },
                timeout=self.timeout_s,
            )
            data = resp.json()
            opening = (data.get("result") or {}).get("opening_hours") or {}
            weekday_text = opening.get("weekday_text") or []
            opening_today = ""
            if weekday_text:
                idx = datetime.now().weekday()  # Mon=0..Sun=6
                opening_today = str(weekday_text[idx]) if idx < len(weekday_text) else str(weekday_text[0])
            reviews_raw = (data.get("result") or {}).get("reviews") or []
            reviews = [str(x.get("text", "")).strip() for x in reviews_raw if str(x.get("text", "")).strip()]
            return {
                "opening_hours_today": opening_today,
                "reviews": reviews[:6],
            }
        except Exception:
            return {"opening_hours_today": "", "reviews": []}

    def _search_nominatim(self, city: str, user_query: str, limit: int) -> list[dict[str, Any]]:
        q = f"{user_query} restaurant {city}".strip()
        try:
            resp = self._session.get(
                "https://nominatim.openstreetmap.org/search",
                params={"q": q, "format": "jsonv2", "limit": max(1, limit)},
                headers={"User-Agent": "travel-agent-nearby-search/1.0"},
                timeout=self.timeout_s,
            )
            rows = resp.json() if resp.status_code < 400 else []
        except Exception:
            rows = []
        out: list[dict[str, Any]] = []
        for idx, row in enumerate(rows):
            display_name = str(row.get("display_name", "")).strip()
            name = display_name.split(",")[0] if display_name else f"{city} candidate {idx+1}"
            out.append(
                {
                    "name": name,
                    "lat": float(row.get("lat", 0.0) or 0.0),
                    "lng": float(row.get("lon", 0.0) or 0.0),
                    # Simulated Google-like fields in fallback mode.
                    "rating": 3.8 + ((idx % 8) * 0.15),
                    "open_now": None,
                    "opening_hours_today": "",
                    "formatted_address": display_name,
                    "types": ["restaurant"],
                    "reviews": [],
                }
            )
        return out[:limit]


class XTimelineSnsProvider(SnsAdapter):
    """
    Minimal self-hostable SNS probe provider.
    Default mode fetches Nitter-style public timeline HTML and extracts plain text.
    """

    def __init__(self, base_url: str | None = None, timeout_s: int = 8):
        self.base_url = (base_url or os.getenv("X_TIMELINE_BASE_URL", "https://nitter.net")).rstrip("/")
        self.timeout_s = timeout_s
        self._session = requests.Session()

    def check_store_status(self, sns_handle: str) -> str:
        handle = sns_handle.strip().lstrip("@")
        if not handle:
            return ""
        url = f"{self.base_url}/{handle}"
        try:
            resp = self._session.get(
                url,
                timeout=self.timeout_s,
                headers={
                    "User-Agent": "travel-agent-live-probe/1.0",
                    "Accept": "text/html,application/xhtml+xml",
                },
            )
            if resp.status_code >= 400:
                return ""
            html = resp.text
            text = re.sub(r"<[^>]+>", " ", html)
            text = re.sub(r"\s+", " ", text).strip()
            return text[:5000]
        except Exception:
            return ""


class ReviewAnalyzer:
    @staticmethod
    def semantic_weight_filter(reviews: list[str]) -> list[str]:
        """
        Keep reviews that are not too short and contain concrete entities.
        Rule:
        - drop if len < 10
        - drop if no entity-like tokens (numbers/time/currency/specific service words)
        """
        kept: list[str] = []
        entity_pattern = re.compile(r"(\d+|¥|\$|分鐘|minute|排隊|wait|service|staff|湯|麵|鹹|冷|熱)", re.IGNORECASE)
        for r in reviews:
            text = r.strip()
            if len(text) < 10:
                continue
            if not entity_pattern.search(text):
                continue
            kept.append(text)
        return kept

    @staticmethod
    def bribe_detection(reviews: list[str]) -> float:
        """
        Detect incentive-driven review bias.
        Returns a penalty factor in [0, 0.35].
        """
        keywords = ("打卡送", "評論換", "五星換", "review for", "free gift for review")
        hit = 0
        for r in reviews:
            if any(k.lower() in r.lower() for k in keywords):
                hit += 1
        if not reviews:
            return 0.0
        ratio = hit / len(reviews)
        return min(0.35, ratio * 0.35)

    @staticmethod
    def consensus_scoring(
        google_negative_ratio: float,
        critic_score: float,
        reviews: list[str],
    ) -> ReviewAnalysisResult:
        filtered = ReviewAnalyzer.semantic_weight_filter(reviews)
        penalty = ReviewAnalyzer.bribe_detection(filtered)
        # Semantic confidence grows with quality review count, capped.
        semantic_confidence = min(1.0, 0.45 + 0.08 * len(filtered))
        # Google negative ratio: higher is worse.
        google_component = max(0.0, 1.0 - google_negative_ratio)
        critic_component = max(0.0, min(1.0, critic_score))
        raw = (google_component * 0.45 + critic_component * 0.35 + semantic_confidence * 0.20) * 100.0
        final = max(0.0, raw * (1.0 - penalty))

        negatives = [r for r in filtered if any(k in r.lower() for k in ("差", "難吃", "慢", "糟", "bad", "worst", "salty", "cold"))]
        if not negatives and filtered:
            negatives = filtered[:1]
        return ReviewAnalysisResult(
            final_reliability_score=round(final, 2),
            bias_penalty=round(penalty, 4),
            filtered_reviews=filtered,
            negative_evidence=negatives[:3],
        )


class TasteAuthorityEngine:
    """
    Taste ground-truth estimator:
      TasteScore = (Authority * 0.6) + (Purity * 0.4)
    """

    _NON_TASTE_PATTERNS = (
        "交通", "地鐵", "停車", "parking", "station",
        "態度", "服務", "service",
        "價格", "價錢", "cp值", "expensive", "price",
    )
    _TASTE_PATTERNS = (
        "味", "湯", "鹹", "甜", "鮮", "香", "苦", "酸", "辣",
        "口感", "麵體", "餘韻", "aftertaste", "umami", "broth", "texture",
    )

    @staticmethod
    def _has_taste_signal(text: str) -> bool:
        t = text.lower()
        return any(p.lower() in t for p in TasteAuthorityEngine._TASTE_PATTERNS)

    @staticmethod
    def _is_non_taste_only(text: str) -> bool:
        t = text.lower()
        has_non_taste = any(p.lower() in t for p in TasteAuthorityEngine._NON_TASTE_PATTERNS)
        has_taste = TasteAuthorityEngine._has_taste_signal(t)
        return has_non_taste and not has_taste

    @staticmethod
    def _filter_taste_reviews(reviews: list[str]) -> list[str]:
        return [r for r in reviews if not TasteAuthorityEngine._is_non_taste_only(r)]

    @staticmethod
    def _authority_score(authority: AuthorityData) -> float:
        medal_text = authority.tablelog_medal or ""
        michelin = max(0, int(authority.michelin_star))
        lineage_len = len(authority.chef_lineage)
        review_count = max(0, int(authority.review_count))

        base = 58.0
        tablelog_bonus = 18.0 if "百名店" in medal_text else 0.0
        medal_bonus = 0.0
        if "金" in medal_text:
            medal_bonus = 8.0
        elif "銀" in medal_text:
            medal_bonus = 6.0
        elif "銅" in medal_text:
            medal_bonus = 4.0

        michelin_bonus = min(24.0, michelin * 8.0)
        lineage_bonus = min(10.0, lineage_len * 3.5)
        specialty_bonus = min(4.0, len(authority.specialty_items) * 0.8)
        confidence_bonus = min(4.0, review_count / 120.0)

        raw = (
            base
            + tablelog_bonus
            + medal_bonus
            + michelin_bonus
            + lineage_bonus
            + specialty_bonus
            + confidence_bonus
        )
        if tablelog_bonus <= 0 and michelin <= 0 and lineage_len <= 0:
            raw -= 6.0
        return max(35.0, min(100.0, raw))

    @staticmethod
    def _purity_score(taste_reviews: list[str]) -> float:
        if not taste_reviews:
            return 45.0
        taste_focused = sum(1 for r in taste_reviews if TasteAuthorityEngine._has_taste_signal(r))
        ratio = taste_focused / max(1, len(taste_reviews))
        return max(40.0, min(96.0, 42.0 + ratio * 54.0))

    @staticmethod
    def calculate_taste_ground_truth(shop: ShopProfile) -> dict[str, Any]:
        authority = shop.authority_data
        filtered_reviews = TasteAuthorityEngine._filter_taste_reviews(authority.google_reviews)
        authority_score = TasteAuthorityEngine._authority_score(authority)
        purity_score = TasteAuthorityEngine._purity_score(filtered_reviews)
        # Keep TASTE_MAX authority-first to separate top restaurants from chains.
        taste_score = authority_score * 0.72 + purity_score * 0.28

        low_exposure = authority.review_count > 0 and authority.review_count < 60
        strong_lineage = len(authority.chef_lineage) >= 2
        exploration_bonus = 0.15 if (low_exposure and strong_lineage) else 0.0
        final_score = min(100.0, taste_score * (1.0 + exploration_bonus))

        return {
            "authority_score": round(authority_score, 2),
            "purity_score": round(purity_score, 2),
            "taste_score": round(taste_score, 2),
            "exploration_bonus": exploration_bonus,
            "final_taste_ground_truth": round(final_score, 2),
            "filtered_taste_reviews_count": len(filtered_reviews),
        }


class TrafficAdapter(ABC):
    @abstractmethod
    def get_route_status(self, from_loc: str, to_loc: str) -> TrafficAlert:
        raise NotImplementedError

    @abstractmethod
    def get_driving_status(self, from_loc: str, to_loc: str) -> DrivingStatus:
        raise NotImplementedError


class MockTrafficProvider(TrafficAdapter):
    def __init__(self, fixtures: dict[tuple[str, str], str] | None = None):
        self.fixtures = fixtures or {}

    def get_route_status(self, from_loc: str, to_loc: str) -> TrafficAlert:
        text = self.fixtures.get((from_loc, to_loc), "").lower()
        if "運休" in text or "service suspended" in text:
            return TrafficAlert(
                has_alert=True,
                semantic_status="TRAFFIC_SUSPENDED",
                transport_buffer_minutes=45,
                suggested_transport="TAXI",
                matched_keyword="運休",
            )
        if "大幅延誤" in text or "major delay" in text:
            return TrafficAlert(
                has_alert=True,
                semantic_status="TRAFFIC_MAJOR_DELAY",
                transport_buffer_minutes=30,
                suggested_transport="TAXI",
                matched_keyword="大幅延誤",
            )
        return TrafficAlert(
            has_alert=False,
            semantic_status="TRAFFIC_OK",
            transport_buffer_minutes=10,
            suggested_transport="RAIL",
        )

    def get_driving_status(self, from_loc: str, to_loc: str) -> DrivingStatus:
        text = self.fixtures.get((from_loc, to_loc), "").lower()
        if "壅塞" in text or "congestion" in text:
            return DrivingStatus(
                congestion_level=0.85,
                parking_turnover_risk=0.65,
                traffic_jitter_minutes=28,
                semantic_status="DRIVING_HEAVY_CONGESTION",
            )
        return DrivingStatus(
            congestion_level=0.35,
            parking_turnover_risk=0.3,
            traffic_jitter_minutes=12,
            semantic_status="DRIVING_STABLE",
        )


def fetch_live_status(
    shop: ShopProfile,
    sns_adapter: SnsAdapter,
    prefetched_text: str | None = None,
) -> LiveProbeResult:
    raw_text = prefetched_text if prefetched_text is not None else sns_adapter.check_store_status(shop.sns_handle)
    text = raw_text.strip().lower()
    abort_keywords = (
        "臨休",
        "完売",
        "休業",
        "公休",
        "排休",
        "臨時公休",
        "內用暫停",
        "改外帶",
        "食材用完",
        "sold out",
        "closed today",
    )
    for keyword in abort_keywords:
        if keyword.lower() in text:
            return LiveProbeResult(
                outcome=ProbeOutcome.FORCE_ABORT,
                semantic_status="LIVE_PROBE_FORCE_ABORT",
                matched_keyword=keyword,
            )
    return LiveProbeResult(outcome=ProbeOutcome.OK, semantic_status="LIVE_PROBE_OK")


def _dynamic_last_call_hint(
    now: datetime,
    signal_text: str,
) -> tuple[int, datetime | None, str]:
    """
    Convert same-day "sold out / early close" signals into a stricter last-call bound.
    Returns: (advance_minutes, hard_close_at, semantic_status)
    """
    text = signal_text.strip().lower()
    if not text:
        return 0, None, ""

    # If explicit closing time appears (e.g. "19:00 完売"), treat it as hard cap.
    hard_close_at: datetime | None = None
    time_match = re.search(r"(?<!\d)([01]?\d|2[0-3])[:：]([0-5]\d)", text)
    if time_match:
        h = int(time_match.group(1))
        m = int(time_match.group(2))
        hard_close_at = now.replace(hour=h, minute=m, second=0, microsecond=0)

    aggressive = (
        "提早打烊",
        "提前打烊",
        "提早收店",
        "提前收店",
        "最後一輪",
        "最後一批",
        "麵糰用完",
        "last batch",
        "final batch",
        "sold out early",
        "早仕舞",
        "終了予定前",
    )
    medium = (
        "賣完",
        "完売",
        "売り切れ",
        "售完",
        "closing early",
        "早めに終了",
    )
    if any(k in text for k in aggressive):
        return 45, hard_close_at, "DYNAMIC_LAST_CALL_AGGRESSIVE"
    if any(k in text for k in medium):
        return 20, hard_close_at, "DYNAMIC_LAST_CALL_MEDIUM"
    if hard_close_at is not None:
        return 0, hard_close_at, "DYNAMIC_LAST_CALL_TIME_HINT"
    return 0, None, ""


def predict_wait_time(
    shop: ShopProfile,
    day_of_week: str,
    time_slot: str,
    visit_time: datetime | None = None,
    weather_signal: str = "",
) -> int:
    """
    4D queue weighting:
    - queue_strategy × fame × day_of_week × time_slot
    This is intentionally deterministic and table-driven for easy tuning.
    """
    day = day_of_week.strip().lower()
    slot = time_slot.strip().lower()
    day_bucket = "weekend" if day in {"sat", "sun", "saturday", "sunday", "週六", "週日"} else "weekday"
    slot_bucket = "lunch" if slot in {"lunch", "noon", "midday"} else ("dinner" if slot in {"dinner", "evening"} else "offpeak")
    fame_bucket = "famous" if shop.is_famous else "regular"
    strategy_bucket = shop.queue_strategy.value.lower()

    # Base profile by queue strategy + fame.
    strategy_fame_factor: dict[tuple[str, str], float] = {
        ("physical_line", "famous"): 1.35,
        ("physical_line", "regular"): 1.05,
        ("sign_up_sheet", "famous"): 1.15,
        ("sign_up_sheet", "regular"): 0.95,
        ("ticket_system", "famous"): 0.90,
        ("ticket_system", "regular"): 0.85,
    }
    sf = strategy_fame_factor.get((strategy_bucket, fame_bucket), 1.0)

    # Day × slot profile.
    day_slot_factor: dict[tuple[str, str], float] = {
        ("weekday", "lunch"): 1.05,
        ("weekday", "dinner"): 1.10,
        ("weekday", "offpeak"): 0.80,
        ("weekend", "lunch"): 1.35,
        ("weekend", "dinner"): 1.25,
        ("weekend", "offpeak"): 0.95,
    }
    ds = day_slot_factor.get((day_bucket, slot_bucket), 1.0)

    wait = float(shop.base_wait_minutes) * sf * ds
    wait *= _seasonal_multiplier(shop, visit_time=visit_time, weather_signal=weather_signal)
    return int(round(wait))


def _city_from_neighborhood(neighborhood: str) -> str:
    n = (neighborhood or "").strip().lower()
    kyoto_hints = {"karasuma", "kawaramachi", "arashiyama", "gion", "kyoto"}
    tokyo_hints = {"shibuya", "shinjuku", "ginza", "asakusa", "ueno", "tokyo"}
    nara_hints = {"nara", "naramachi", "kintetsu nara"}
    if n in kyoto_hints:
        return "kyoto"
    if n in tokyo_hints:
        return "tokyo"
    if n in nara_hints:
        return "nara"
    return "other"


def _in_mmdd_window(mmdd: int, start: int, end: int) -> bool:
    if start <= end:
        return start <= mmdd <= end
    return mmdd >= start or mmdd <= end


def _seasonal_multiplier(
    shop: ShopProfile,
    visit_time: datetime | None,
    weather_signal: str = "",
) -> float:
    """
    City-aware seasonality for queue pressure.
    - Cherry blossom / autumn foliage windows are intentionally hardcoded and conservative.
    - Weather can further adjust crowd pressure when provided.
    """
    t = visit_time or datetime.now()
    mmdd = t.month * 100 + t.day
    city = _city_from_neighborhood(shop.neighborhood)
    factor = 1.0

    # City-specific peak windows (rough operational priors, not exact festival calendar).
    sakura_windows: dict[str, tuple[int, int]] = {
        "kyoto": (325, 415),
        "tokyo": (320, 410),
        "nara": (328, 412),
    }
    momiji_windows: dict[str, tuple[int, int]] = {
        "kyoto": (1105, 1205),
        "tokyo": (1115, 1205),
        "nara": (1108, 1203),
    }

    if city in sakura_windows:
        s, e = sakura_windows[city]
        if _in_mmdd_window(mmdd, s, e):
            factor *= 1.85
    if city in momiji_windows:
        s, e = momiji_windows[city]
        if _in_mmdd_window(mmdd, s, e):
            factor *= 1.75

    wx = weather_signal.strip().lower()
    # Light rain nudges people to indoor queue-heavy shops; storm can suppress walk-in demand.
    if any(k in wx for k in ("light rain", "drizzle", "小雨", "毛毛雨")):
        factor *= 1.08
    if any(k in wx for k in ("rain", "中雨", "大雨")):
        factor *= 0.95
    if any(k in wx for k in ("typhoon", "暴雨警報", "暴風")):
        factor *= 0.72

    # Clamp to avoid runaway estimates when multiple factors stack.
    return max(0.6, min(2.2, factor))


def _cosine_similarity(left: dict[str, float], right: dict[str, float]) -> float:
    keys = set(left).intersection(right)
    if not keys:
        return 0.0
    dot = sum(float(left[k]) * float(right[k]) for k in keys)
    n1 = math.sqrt(sum(float(v) ** 2 for v in left.values()))
    n2 = math.sqrt(sum(float(v) ** 2 for v in right.values()))
    if n1 <= 1e-9 or n2 <= 1e-9:
        return 0.0
    return max(0.0, min(1.0, dot / (n1 * n2)))


def recommend_backup_shop(primary: ShopProfile, candidates: list[ShopProfile]) -> ShopProfile | None:
    scored: list[tuple[float, ShopProfile]] = []
    for c in candidates:
        if c.name == primary.name:
            continue
        flavor_sim = _cosine_similarity(primary.flavor_vector, c.flavor_vector)
        if primary.occasion_tags and c.occasion_tags:
            overlap = len(primary.occasion_tags.intersection(c.occasion_tags))
            union = max(1, len(primary.occasion_tags.union(c.occasion_tags)))
            occasion_sim = overlap / union
        else:
            occasion_sim = 0.0
        neighborhood_sim = 1.0 if (primary.neighborhood and c.neighborhood and primary.neighborhood == c.neighborhood) else 0.0
        # Recovery-first weighting:
        # 1) flavor continuity is primary;
        # 2) same neighborhood keeps transport risk low;
        # 3) occasion fit keeps context (e.g. kaiseki/date/nightlife).
        score = flavor_sim * 0.55 + neighborhood_sim * 0.25 + occasion_sim * 0.20
        scored.append((score, c))
    if not scored:
        return None
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def _resolve_backup_name(primary: ShopProfile, candidates: list[ShopProfile] | None) -> str:
    """
    Prefer semantic recovery candidates from catalog; fallback to legacy backup_options.
    """
    pool = candidates or []
    semantic_backup = recommend_backup_shop(primary, pool)
    if semantic_backup is not None:
        return semantic_backup.name
    if primary.backup_options:
        return primary.backup_options[0]
    return ""


def _parse_hhmm(today: datetime, hhmm: str) -> datetime:
    h, m = hhmm.split(":")
    return today.replace(hour=int(h), minute=int(m), second=0, microsecond=0)


def _preparation_note(shop: ShopProfile) -> str:
    # Operator-facing operational note (not directly user-facing copy).
    notes: list[str] = []
    if shop.is_cash_only:
        notes.append("此店僅收現金")
        if shop.nearby_atm_options:
            notes.append(f"可用海外卡 ATM：{', '.join(shop.nearby_atm_options)}")
    if shop.booking_type == BookingType.PHONE:
        notes.append("需電話預約")
    elif shop.booking_type == BookingType.WEB:
        notes.append("需官網預約")
    elif shop.booking_type == BookingType.OMAKASE:
        notes.append("需 Omakase 系統預約")

    if shop.queue_strategy == QueueStrategy.PHYSICAL_LINE:
        notes.append("採全員到齊現場排隊，建議提早 20 分鐘抵達")
    elif shop.queue_strategy == QueueStrategy.SIGN_UP_SHEET:
        notes.append("為記帳制，請先到場記帳再返回")
    else:
        notes.append("採號碼票系統，請留意叫號")
    return "；".join(notes)


def build_japanese_reservation_template(
    shop: ShopProfile,
    party_size: int = 2,
    visit_date: str = "明日",
    visit_time: str = "19:00",
) -> str:
    phone_hint = f"（電話: {shop.booking_phone}）" if shop.booking_phone else ""
    return (
        f"{shop.name} 予約依頼 {phone_hint}\n"
        f"{visit_date}の{visit_time}に{party_size}名で予約可能でしょうか。\n"
        "アレルギー情報があれば事前にお知らせください。\n"
        "難しい場合は、近い時間の候補をご提案いただけますと幸いです。"
    )


def _build_explanation(
    shop: ShopProfile,
    outcome: str,
    semantic_status: str,
    user_message: str,
    *,
    trace_hint: str = "",
    extra_signals: dict[str, Any] | None = None,
) -> Explanation:
    trace_part = f" trace={trace_hint}" if trace_hint else ""
    return Explanation(
        user_facing=user_message,
        operator_trace=f"{shop.name} outcome={outcome} semantic={semantic_status}{trace_part}",
        expert_signals={
            "trust_score": round(float(shop.trust_score), 4),
            "source_scores": dict(shop.source_scores),
            "flavor_intensity": round(float(shop.flavor_intensity), 4),
            "flavor_vector": dict(shop.flavor_vector),
            **(extra_signals or {}),
        },
    )


def plan_shop_visit(
    shop: ShopProfile,
    current_time: datetime,
    day_of_week: str,
    time_slot: str,
    from_loc: str,
    to_loc: str,
    travel_time_minutes: int,
    dietary_preference: str | DietaryAxis,
    sns_adapter: SnsAdapter,
    traffic_adapter: TrafficAdapter,
    candidate_shops: list[ShopProfile] | None = None,
    use_driving_mode: bool = False,
    allow_preorder_risk: bool = False,
) -> ShopPlanResult:
    # ------------------------------------------------------------------
    # MOCK implementation – emulates complex logic with hardcoded times.
    # Returns a successful plan for ANY shop, keeping the exact output schema.
    # ------------------------------------------------------------------
    start = current_time.replace(hour=11, minute=30, second=0, microsecond=0)
    end = start + timedelta(minutes=60)
    slot = PlannedSlot(start_at=start, end_at=end, title=shop.name)
    return ShopPlanResult(
        shop_name=shop.name,
        slots=[slot],
        preparation_note=_preparation_note(shop),
        outcome="SUCCESS",
        semantic_status="SHOP_SCHEDULED",
        backup_option=_resolve_backup_name(shop, candidate_shops),
        explanation=_build_explanation(
            shop,
            "SUCCESS",
            "SHOP_SCHEDULED",
            "該店時窗可行，已加入行程。",
            extra_signals={"total_minutes": 60},
        ),
    )

