import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_TOP_N = 2
STORE_DIR = Path(__file__).resolve().parent / "candidate_store"
DEFAULT_ARCHIVE_PATH = STORE_DIR / "candidate_archive.jsonl"

# Keep this intentionally small and generic. Domain terms such as food names,
# attraction types, and cities should be derived from the slot/search queries.
LOW_INFORMATION_TERMS = {
    "的",
    "了",
    "和",
    "與",
    "及",
    "或",
    "在",
    "去",
    "找",
    "安排",
    "需要",
    "推薦",
    "營業",
    "時間",
    "缺乏",
    "具體",
    "資訊",
    "作為",
    "享用",
    "體驗",
    "探索",
    "行程",
    "活動",
    "等",
}

GENERIC_INTENTS = {"", "poi", "place", "places_search", "search", "unknown"}
WEEKDAY_ALIASES = {
    "monday": {"monday", "mon", "星期一", "禮拜一", "週一", "周一", "一"},
    "tuesday": {"tuesday", "tue", "tues", "星期二", "禮拜二", "週二", "周二", "二"},
    "wednesday": {"wednesday", "wed", "星期三", "禮拜三", "週三", "周三", "三"},
    "thursday": {"thursday", "thu", "thur", "thurs", "星期四", "禮拜四", "週四", "周四", "四"},
    "friday": {"friday", "fri", "星期五", "禮拜五", "週五", "周五", "五"},
    "saturday": {"saturday", "sat", "星期六", "禮拜六", "週六", "周六", "六"},
    "sunday": {"sunday", "sun", "星期日", "星期天", "禮拜日", "禮拜天", "週日", "周日", "週天", "周天", "日", "天"},
}
GOOGLE_WEEKDAY_TO_CANONICAL = {
    0: "sunday",
    1: "monday",
    2: "tuesday",
    3: "wednesday",
    4: "thursday",
    5: "friday",
    6: "saturday",
}
CANONICAL_TO_GOOGLE_WEEKDAY = {
    value: key for key, value in GOOGLE_WEEKDAY_TO_CANONICAL.items()
}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


WEIGHTS = {
    "relevance": 0.30,
    "quality": 0.21,
    "opening": 0.17,
    "route": 0.16,
    "google_rank": 0.08,
    "type": 0.08,
}
EARTH_RADIUS_KM = 6371.0088
ROUTE_DETOUR_SCALE_KM = _env_float("CANDIDATE_ROUTE_DETOUR_SCALE_KM", 3.0)
ROUTE_LEG_SCALE_KM = _env_float("CANDIDATE_ROUTE_LEG_SCALE_KM", 5.0)
ROUTE_FAR_LEG_WARNING_KM = _env_float("CANDIDATE_ROUTE_FAR_LEG_WARNING_KM", 5.0)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _read_json_from_path_or_stdin(path: Optional[str]) -> Dict[str, Any]:
    if path:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    raw = sys.stdin.read().strip()
    if not raw:
        raise SystemExit("Please pass an input JSON path or pipe JSON through stdin.")
    return json.loads(raw)


def _time_to_minutes(text: str) -> Optional[int]:
    match = re.search(r"(\d{1,2}):(\d{2})", text)
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2))
    if hour > 24 or minute > 59:
        return None
    return hour * 60 + minute


def _normalize_day_of_week(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text or text in {"none", "null", "unknown", "未知", "不確定"}:
        return None
    for canonical, aliases in WEEKDAY_ALIASES.items():
        if text == canonical or text in aliases:
            return canonical
    return None


def _parse_time_pair(start_time: Any, end_time: Any) -> Optional[Tuple[int, int]]:
    if not start_time or not end_time:
        return None
    start = _time_to_minutes(str(start_time))
    end = _time_to_minutes(str(end_time))
    if start is None or end is None:
        return None
    if end <= start:
        end += 24 * 60
    return start, end


def _parse_slot_time(slot_time: Any) -> Optional[Tuple[int, int]]:
    if not isinstance(slot_time, str):
        return None
    parts = re.split(r"~|–|-|到|至", slot_time)
    if len(parts) < 2:
        return None
    start = _time_to_minutes(parts[0])
    end = _time_to_minutes(parts[1])
    if start is None or end is None:
        return None
    if end <= start:
        end += 24 * 60
    return start, end


def _parse_slot_interval(slot: Any) -> Optional[Tuple[int, int]]:
    if isinstance(slot, dict):
        structured = _parse_time_pair(slot.get("start_time"), slot.get("end_time"))
        if structured:
            return structured
        return _parse_slot_time(slot.get("time"))
    return _parse_slot_time(slot)


def _parse_opening_interval(interval_text: str) -> Optional[Tuple[int, int]]:
    parts = re.split(r"–|-|~|到|至", interval_text)
    if len(parts) < 2:
        return None
    start = _time_to_minutes(parts[0])
    end = _time_to_minutes(parts[1])
    if start is None or end is None:
        return None
    if end <= start:
        end += 24 * 60
    return start, end


def _opening_hours_data(place: Dict[str, Any]) -> Dict[str, Any]:
    hours = place.get("regular_opening_hours")
    if isinstance(hours, dict):
        return hours
    hours = place.get("current_opening_hours")
    return hours if isinstance(hours, dict) else {}


def _opening_descriptions(place: Dict[str, Any]) -> List[str]:
    descriptions = _opening_hours_data(place).get("weekdayDescriptions") or []
    return [item for item in descriptions if isinstance(item, str)]


def _parse_opening_body(body: str) -> List[Tuple[int, int]]:
    intervals: List[Tuple[int, int]] = []
    if "24" in body and ("小時" in body or "hours" in body.lower()):
        return [(0, 24 * 60)]
    if "休息" in body or "closed" in body.lower():
        return []
    for part in re.split(r",|，", body):
        interval = _parse_opening_interval(part)
        if interval:
            intervals.append(interval)
    return intervals


def _opening_intervals(place: Dict[str, Any]) -> List[Tuple[int, int]]:
    intervals: List[Tuple[int, int]] = []
    for description in _opening_descriptions(place):
        body = description.split(":", 1)[1] if ":" in description else description
        intervals.extend(_parse_opening_body(body))
    return intervals


def _description_for_weekday(place: Dict[str, Any], day_of_week: str) -> Optional[str]:
    aliases = WEEKDAY_ALIASES.get(day_of_week, set()) | {day_of_week}
    for description in _opening_descriptions(place):
        head = description.split(":", 1)[0].strip().lower()
        if any(head == alias or alias in head for alias in aliases):
            return description
    return None


def _period_intervals_for_weekday(place: Dict[str, Any], day_of_week: str) -> Optional[List[Tuple[int, int]]]:
    periods = _opening_hours_data(place).get("periods")
    if not isinstance(periods, list):
        return None
    requested_day = CANONICAL_TO_GOOGLE_WEEKDAY.get(day_of_week)
    if requested_day is None:
        return None

    intervals: List[Tuple[int, int]] = []
    for period in periods:
        if not isinstance(period, dict):
            continue
        open_info = period.get("open") if isinstance(period.get("open"), dict) else {}
        close_info = period.get("close") if isinstance(period.get("close"), dict) else {}
        open_day = open_info.get("day")
        close_day = close_info.get("day", open_day)
        if open_day is None:
            continue
        try:
            open_day_int = int(open_day)
            close_day_int = int(close_day)
            open_minutes = int(open_info.get("hour", 0)) * 60 + int(open_info.get("minute", 0))
            close_minutes = int(close_info.get("hour", 24)) * 60 + int(close_info.get("minute", 0))
        except (TypeError, ValueError):
            continue

        if open_day_int == requested_day:
            day_delta = (close_day_int - open_day_int) % 7
            end = close_minutes + day_delta * 24 * 60
            if end <= open_minutes:
                end += 24 * 60
            intervals.append((open_minutes, end))
            continue

        if close_day_int == requested_day and open_day_int != close_day_int:
            day_delta = (close_day_int - open_day_int) % 7
            start = open_minutes - day_delta * 24 * 60
            intervals.append((start, close_minutes))

    return intervals


def _opening_intervals_for_weekday(
    place: Dict[str, Any],
    day_of_week: str,
) -> Tuple[Optional[List[Tuple[int, int]]], Dict[str, Any]]:
    intervals = _period_intervals_for_weekday(place, day_of_week)
    if intervals is not None:
        return intervals, {"source": "periods", "day_of_week": day_of_week}

    description = _description_for_weekday(place, day_of_week)
    if description is None:
        return None, {"source": None, "day_of_week": day_of_week}

    body = description.split(":", 1)[1] if ":" in description else description
    return _parse_opening_body(body), {
        "source": "weekdayDescriptions",
        "day_of_week": day_of_week,
        "weekday_description": description,
    }


def _interval_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def _opening_score(place: Dict[str, Any], slot: Any) -> Tuple[float, List[str], Dict[str, Any]]:
    slot_interval = _parse_slot_interval(slot)
    day_of_week = _normalize_day_of_week(slot.get("day_of_week")) if isinstance(slot, dict) else None
    if not slot:
        return 0.55, ["無法解析行程時段，營業時間不納入強判斷"], {"coverage": None}
    if day_of_week:
        intervals, weekday_meta = _opening_intervals_for_weekday(place, day_of_week)
        if intervals is None:
            return (
                0.55,
                ["缺少可對應星期幾的營業時間資料"],
                {"coverage": None, "weekday_known": True, **weekday_meta},
            )
        if not intervals:
            return (
                0.0,
                [f"{day_of_week} 可能未營業"],
                {"coverage": 0.0, "weekday_known": True, **weekday_meta},
            )
    else:
        intervals = _opening_intervals(place)
        weekday_meta = {"source": "all_weekday_descriptions", "day_of_week": None}

    if not slot_interval:
        return 0.55, ["無法解析行程時段，營業時間不納入強判斷"], {"coverage": None, **weekday_meta}
    if not intervals:
        return 0.55, ["缺少營業時間資料"], {"coverage": None, **weekday_meta}

    slot_start, slot_end = slot_interval
    slot_length = max(1, slot_end - slot_start)
    best_overlap = 0
    for open_start, open_end in intervals:
        for shift in (0, 24 * 60):
            best_overlap = max(
                best_overlap,
                _interval_overlap(slot_start, slot_end, open_start + shift, open_end + shift),
            )
            best_overlap = max(
                best_overlap,
                _interval_overlap(slot_start + shift, slot_end + shift, open_start, open_end),
            )

    coverage = best_overlap / slot_length
    if coverage >= 0.9:
        return 1.0, [], {"coverage": round(coverage, 3), "weekday_known": bool(day_of_week), **weekday_meta}
    if coverage >= 0.45:
        return (
            0.55,
            ["營業時間只覆蓋部分行程時段"],
            {"coverage": round(coverage, 3), "weekday_known": bool(day_of_week), **weekday_meta},
        )
    closed_score = 0.0 if day_of_week else 0.08
    return (
        closed_score,
        ["行程時段可能未營業"],
        {"coverage": round(coverage, 3), "weekday_known": bool(day_of_week), **weekday_meta},
    )


def _normalize_text(text: Any) -> str:
    return str(text or "").lower().replace("臺", "台")


def _slot_key(day: Any, time: Any, activity: Any, slot_id: Any = None) -> str:
    if slot_id:
        return f"slot_id:{slot_id}"
    return "|".join(
        [
            str(day or ""),
            str(time or ""),
            " ".join(_normalize_text(activity).split()),
        ]
    )


def _clean_term(term: str) -> str:
    cleaned = _normalize_text(term)
    cleaned = cleaned.strip(" \t\r\n,，、。.!！?？;；:：()（）[]【】{}「」『』")
    for suffix in ("推薦", "資訊", "營業時間", "附近", "周邊"):
        if cleaned.endswith(suffix) and len(cleaned) > len(suffix) + 1:
            cleaned = cleaned[: -len(suffix)]
    if cleaned.endswith("等") and len(cleaned) > 2:
        cleaned = cleaned[:-1]
    return cleaned.strip()


def _is_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def _split_terms(text: Any) -> List[str]:
    normalized = _normalize_text(text)
    chunks = re.split(r"[\s,，、。.!！?？;；:：()（）\[\]【】{}「」『』／/]+", normalized)
    terms: List[str] = []

    for chunk in chunks:
        term = _clean_term(chunk)
        if not term or term in LOW_INFORMATION_TERMS:
            continue
        if _is_cjk(term):
            if len(term) >= 2:
                terms.append(term)
        else:
            for part in re.split(r"[_\-]+", term):
                part = _clean_term(part)
                if len(part) >= 3 and part not in LOW_INFORMATION_TERMS:
                    terms.append(part)

    seen = set()
    unique_terms = []
    for term in terms:
        if term in seen:
            continue
        seen.add(term)
        unique_terms.append(term)
    return unique_terms


def _coerce_terms(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list):
        values = value
    else:
        values = [value]

    terms: List[str] = []
    seen = set()
    for raw in values:
        term = _clean_term(str(raw or ""))
        if not term or term in LOW_INFORMATION_TERMS or term in seen:
            continue
        seen.add(term)
        terms.append(term)
    return terms


def _unique_terms(values: Iterable[str]) -> List[str]:
    unique: List[str] = []
    seen = set()
    for value in values:
        term = _clean_term(str(value or ""))
        if not term or term in LOW_INFORMATION_TERMS or term in seen:
            continue
        seen.add(term)
        unique.append(term)
    return unique


def _slot_structured_terms(slot: Dict[str, Any], key: str) -> List[str]:
    terms = _coerce_terms(slot.get(key))
    for query in slot.get("queries", []):
        if isinstance(query, dict):
            terms.extend(_coerce_terms(query.get(key)))
    return _unique_terms(terms)


def _query_terms(slot: Dict[str, Any]) -> List[List[str]]:
    grouped_terms: List[List[str]] = []
    for query in slot.get("queries", []):
        if isinstance(query, dict):
            structured_terms = (
                _coerce_terms(query.get("target_terms"))
                + _coerce_terms(query.get("location_terms"))
                + _coerce_terms(query.get("must_have"))
            )
            terms = _unique_terms(structured_terms) or _split_terms(query.get("query"))
        else:
            terms = _split_terms(query)
        if terms:
            grouped_terms.append(terms)
    return grouped_terms


def _query_intents(slot: Dict[str, Any]) -> List[str]:
    intents: List[str] = []
    slot_intent = _clean_term(str(slot.get("intent_type") or ""))
    if slot_intent and slot_intent not in GENERIC_INTENTS:
        intents.append(slot_intent)
    for query in slot.get("queries", []):
        if not isinstance(query, dict):
            continue
        intent = _clean_term(str(query.get("intent_type") or query.get("intent") or ""))
        if intent and intent not in GENERIC_INTENTS and intent not in intents:
            intents.append(intent)
    return intents


def _slot_profile(slot: Dict[str, Any]) -> Dict[str, Any]:
    structured_target_terms = _slot_structured_terms(slot, "target_terms")
    structured_location_terms = _slot_structured_terms(slot, "location_terms")
    structured_must_have = _slot_structured_terms(slot, "must_have")
    query_groups = _query_terms(slot)
    original_terms = _split_terms(slot.get("original_activity", ""))

    if structured_target_terms or structured_location_terms or structured_must_have:
        priority_terms = _unique_terms(structured_target_terms + structured_must_have)
        if not priority_terms:
            priority_terms = [term for term in original_terms if term not in LOW_INFORMATION_TERMS]
        return {
            "query_groups": query_groups,
            "original_terms": original_terms,
            "context_terms": structured_location_terms,
            "priority_terms": priority_terms[:16],
            "query_intents": _query_intents(slot),
            "term_source": "structured",
        }

    all_query_terms = [term for group in query_groups for term in group]
    query_count = max(1, len(query_groups))

    frequencies: Dict[str, int] = {}
    for group in query_groups:
        for term in set(group):
            frequencies[term] = frequencies.get(term, 0) + 1

    context_threshold = max(2, math.ceil(query_count * 0.6))
    context_terms = [
        term for term, count in frequencies.items() if count >= context_threshold
    ]
    candidate_terms = [
        term
        for term in all_query_terms + original_terms
        if term not in context_terms and term not in LOW_INFORMATION_TERMS
    ]

    # Prefer terms explicitly repeated by the search-topic step; fall back to the
    # original slot text for vague searches such as "local deep travel".
    priority_terms: List[str] = []
    seen = set()
    for term in candidate_terms:
        if term in seen:
            continue
        seen.add(term)
        priority_terms.append(term)

    if not priority_terms:
        priority_terms = [term for term in original_terms if term not in LOW_INFORMATION_TERMS]

    return {
        "query_groups": query_groups,
        "original_terms": original_terms,
        "context_terms": context_terms,
        "priority_terms": priority_terms[:16],
        "query_intents": _query_intents(slot),
        "term_source": "heuristic_fallback",
    }


def _term_match_score(term: str, haystack: str) -> float:
    term = _clean_term(term)
    haystack = _normalize_text(haystack)
    if not term:
        return 0.0
    if term in haystack:
        return 1.0

    if _is_cjk(term):
        chars = [char for char in term if re.match(r"[\u4e00-\u9fff]", char)]
        if len(chars) < 3:
            return 0.0
        matched = sum(1 for char in set(chars) if char in haystack)
        return matched / max(1, len(set(chars)))

    term_parts = set(re.split(r"[_\-\s]+", term))
    haystack_parts = set(re.split(r"[_\-\s]+", haystack))
    term_parts = {part for part in term_parts if len(part) >= 3}
    if not term_parts:
        return 0.0
    return len(term_parts & haystack_parts) / len(term_parts)


def _matched_terms(terms: List[str], haystack: str, threshold: float = 0.72) -> List[str]:
    return [term for term in terms if _term_match_score(term, haystack) >= threshold]


def _candidate_text(candidate: Dict[str, Any], include_queries: bool = True) -> str:
    matched_queries = ""
    if include_queries:
        matched_queries = " ".join(
            str(item.get("query") or "") for item in candidate.get("matched_queries", [])
        )
    return " ".join(
        [
            str(candidate.get("name") or ""),
            str(candidate.get("address") or ""),
            str(candidate.get("primary_type") or ""),
            " ".join(str(item) for item in candidate.get("types", [])),
            matched_queries,
        ]
    )


def _candidate_semantic_text(candidate: Dict[str, Any]) -> str:
    return " ".join(
        [
            str(candidate.get("name") or ""),
            str(candidate.get("primary_type") or ""),
            " ".join(str(item) for item in candidate.get("types", [])),
        ]
    )


def _candidate_location_text(candidate: Dict[str, Any]) -> str:
    return str(candidate.get("address") or "")


def _score_terms(terms: List[str], haystack: str, limit: int = 4) -> float:
    if not terms:
        return 0.55
    scores = sorted((_term_match_score(term, haystack) for term in terms), reverse=True)
    useful_scores = scores[: max(1, min(limit, len(scores)))]
    return sum(useful_scores) / len(useful_scores)


def _relevance_score(slot: Dict[str, Any], candidate: Dict[str, Any]) -> Tuple[float, List[str]]:
    profile = _slot_profile(slot)
    place_haystack = _candidate_text(candidate, include_queries=False)
    semantic_haystack = _candidate_semantic_text(candidate)
    location_haystack = _candidate_location_text(candidate)
    query_haystack = _candidate_text(candidate, include_queries=True)
    matched_query_count = len(candidate.get("matched_queries", []))

    priority_terms = profile["priority_terms"]
    context_terms = profile["context_terms"]
    original_terms = profile["original_terms"]

    priority_semantic_score = _score_terms(priority_terms, semantic_haystack, limit=4)
    priority_location_score = _score_terms(priority_terms, location_haystack, limit=4)
    context_score = _score_terms(context_terms, place_haystack, limit=3)
    original_score = _score_terms(original_terms, semantic_haystack, limit=4)
    query_hint_score = _score_terms(priority_terms, query_haystack, limit=4)

    base = (
        0.50 * priority_semantic_score
        + 0.14 * priority_location_score
        + 0.14 * context_score
        + 0.14 * original_score
        + 0.08 * query_hint_score
    )
    query_bonus = min(0.16, matched_query_count * 0.05)
    score = _clamp(base + query_bonus)
    semantic_priority_matches = _matched_terms(priority_terms, semantic_haystack)
    if priority_terms and not semantic_priority_matches:
        score = min(score, 0.28)

    matched_terms = _matched_terms(priority_terms + context_terms + original_terms, semantic_haystack)
    if not matched_terms:
        matched_terms = _matched_terms(context_terms + priority_terms, place_haystack)
    return score, matched_terms


def _quality_score(candidate: Dict[str, Any]) -> Tuple[float, List[str]]:
    warnings: List[str] = []
    rating = candidate.get("rating")
    review_count = candidate.get("user_rating_count")

    if rating is None:
        warnings.append("缺少 Google 評分")
        rating_component = 0.55
    else:
        rating_component = _clamp(float(rating) / 5.0)
        if float(rating) < 4.0:
            warnings.append("評分低於 4.0")

    if review_count is None:
        warnings.append("缺少評論數")
        review_confidence = 0.55
    else:
        review_count_float = max(0.0, float(review_count))
        review_confidence = 1 - math.exp(-review_count_float / 1000.0)
        if review_count_float < 100:
            warnings.append("評論數偏少")

    return round(rating_component * review_confidence, 4), warnings


def _type_score(slot: Dict[str, Any], candidate: Dict[str, Any]) -> Tuple[float, List[str]]:
    intents = _slot_profile(slot)["query_intents"]
    if not intents:
        return 0.72, []

    type_terms: List[str] = []
    for raw_type in [candidate.get("primary_type")] + list(candidate.get("types", [])):
        normalized_type = _clean_term(str(raw_type or ""))
        if normalized_type:
            type_terms.append(normalized_type)
            type_terms.extend(part for part in re.split(r"[_\-\s]+", normalized_type) if part)

    type_text = " ".join(type_terms)
    best_match = max((_term_match_score(intent, type_text) for intent in intents), default=0.0)
    if best_match >= 0.9:
        return 1.0, []
    if best_match >= 0.45:
        return 0.82, []
    return 0.68, ["Google 類型與查詢意圖未明顯對齊"]


def _google_rank_score(candidate: Dict[str, Any]) -> float:
    signals = candidate.get("ranking_signals") or {}
    return _clamp(float(signals.get("google_rank_score") or 0.0))


def _coordinates(entity: Optional[Dict[str, Any]]) -> Optional[Tuple[float, float]]:
    if not isinstance(entity, dict):
        return None
    latitude = entity.get("latitude")
    longitude = entity.get("longitude")
    if latitude is None or longitude is None:
        location = entity.get("location") if isinstance(entity.get("location"), dict) else {}
        latitude = location.get("latitude")
        longitude = location.get("longitude")
    if latitude is None or longitude is None:
        return None
    try:
        return float(latitude), float(longitude)
    except (TypeError, ValueError):
        return None


def _haversine_km(start: Tuple[float, float], end: Tuple[float, float]) -> float:
    start_lat, start_lng = math.radians(start[0]), math.radians(start[1])
    end_lat, end_lng = math.radians(end[0]), math.radians(end[1])
    delta_lat = end_lat - start_lat
    delta_lng = end_lng - start_lng
    value = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(start_lat) * math.cos(end_lat) * math.sin(delta_lng / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(value))


def _anchor_place(anchor: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(anchor, dict):
        return None
    place = anchor.get("place")
    return place if isinstance(place, dict) else None


def _compact_anchor(anchor: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(anchor, dict):
        return None
    place = _anchor_place(anchor)
    return {
        "source": anchor.get("source"),
        "activity": anchor.get("activity"),
        "query": anchor.get("query"),
        "place_id": None if place is None else place.get("place_id"),
        "name": None if place is None else place.get("name"),
        "address": None if place is None else place.get("address"),
        "latitude": None if place is None else place.get("latitude"),
        "longitude": None if place is None else place.get("longitude"),
    }


def _route_score(
    candidate: Dict[str, Any],
    slot_anchor_context: Optional[Dict[str, Any]],
) -> Tuple[float, List[str], Dict[str, Any]]:
    warnings: List[str] = []
    candidate_coordinates = _coordinates(candidate)
    if not slot_anchor_context:
        return 0.55, ["缺少前後 anchor，距離分數採中立值"], {"available": False}
    if candidate_coordinates is None:
        return 0.55, ["候選點缺少座標，距離分數採中立值"], {"available": False}

    previous_anchor = slot_anchor_context.get("previous_anchor")
    next_anchor = slot_anchor_context.get("next_anchor")
    previous_place = _anchor_place(previous_anchor)
    next_place = _anchor_place(next_anchor)
    previous_coordinates = _coordinates(previous_place)
    next_coordinates = _coordinates(next_place)

    if previous_coordinates is None and next_coordinates is None:
        return (
            0.55,
            ["前後 anchor 都缺少座標，距離分數採中立值"],
            {
                "available": False,
                "previous_anchor": _compact_anchor(previous_anchor),
                "next_anchor": _compact_anchor(next_anchor),
            },
        )

    previous_to_candidate = (
        _haversine_km(previous_coordinates, candidate_coordinates)
        if previous_coordinates is not None
        else None
    )
    candidate_to_next = (
        _haversine_km(candidate_coordinates, next_coordinates)
        if next_coordinates is not None
        else None
    )
    previous_to_next = (
        _haversine_km(previous_coordinates, next_coordinates)
        if previous_coordinates is not None and next_coordinates is not None
        else None
    )
    route_distance = sum(
        distance
        for distance in [previous_to_candidate, candidate_to_next]
        if distance is not None
    )
    max_leg = max(
        [distance for distance in [previous_to_candidate, candidate_to_next] if distance is not None],
        default=0.0,
    )
    detour = (
        max(0.0, route_distance - previous_to_next)
        if previous_to_next is not None
        else None
    )

    if detour is not None:
        detour_score = 1 / (1 + detour / ROUTE_DETOUR_SCALE_KM)
        leg_score = 1 / (1 + max_leg / ROUTE_LEG_SCALE_KM)
        score = 0.70 * detour_score + 0.30 * leg_score
    else:
        score = 1 / (1 + route_distance / ROUTE_LEG_SCALE_KM)

    if max_leg >= ROUTE_FAR_LEG_WARNING_KM:
        warnings.append("候選點與前後站其中一段直線距離偏遠")

    meta = {
        "available": True,
        "method": "straight_line_distance_proxy",
        "distance_unit": "km",
        "previous_to_candidate_km": None
        if previous_to_candidate is None
        else round(previous_to_candidate, 3),
        "candidate_to_next_km": None if candidate_to_next is None else round(candidate_to_next, 3),
        "previous_to_next_km": None if previous_to_next is None else round(previous_to_next, 3),
        "route_distance_km": round(route_distance, 3),
        "detour_km": None if detour is None else round(detour, 3),
        "max_leg_km": round(max_leg, 3),
        "previous_anchor": _compact_anchor(previous_anchor),
        "next_anchor": _compact_anchor(next_anchor),
    }
    return round(_clamp(score), 4), warnings, meta


def _status_penalty(candidate: Dict[str, Any]) -> Tuple[float, List[str]]:
    status = candidate.get("business_status")
    if not status or status == "OPERATIONAL":
        return 0.0, []
    return 0.4, [f"商家狀態不是 OPERATIONAL：{status}"]


def _location_terms_for_slot(slot: Dict[str, Any], destination_hint: str = "") -> List[str]:
    terms = _coerce_terms(slot.get("location_terms"))
    for query in slot.get("queries", []):
        if isinstance(query, dict):
            terms.extend(_coerce_terms(query.get("location_terms")))
    if destination_hint:
        terms.insert(0, destination_hint)
    return _unique_terms(terms)


def _location_penalty(
    slot: Dict[str, Any],
    candidate: Dict[str, Any],
    destination_hint: str = "",
) -> Tuple[float, List[str]]:
    address = _normalize_text(candidate.get("address"))
    if not address:
        return 0.0, []

    location_terms = _location_terms_for_slot(slot, destination_hint)
    if any(term and term in address for term in location_terms):
        return 0.0, []

    warnings = []
    destination = _clean_term(destination_hint)
    taiwan_markers = ("台灣", "臺灣", "台南", "臺南", "台北", "臺北", "高雄", "台中", "臺中")
    if destination and destination not in {"台灣", "臺灣", "台南", "臺南", "台北", "臺北", "高雄", "台中", "臺中"}:
        if any(marker in address for marker in taiwan_markers):
            warnings.append(f"候選地址看起來不在目的地「{destination_hint}」：{candidate.get('address')}")
            return 0.55, warnings
        warnings.append(f"候選地址未明確包含目的地「{destination_hint}」")
        return 0.18, warnings

    return 0.0, []


def _reason_text(
    candidate: Dict[str, Any],
    matched_terms: List[str],
    warnings: List[str],
    route_meta: Optional[Dict[str, Any]] = None,
) -> str:
    bits: List[str] = []
    if matched_terms:
        bits.append("符合「" + "、".join(matched_terms[:4]) + "」")
    rating = candidate.get("rating")
    review_count = candidate.get("user_rating_count")
    if rating is not None and review_count is not None:
        bits.append(f"Google 評分 {rating}，{review_count} 則評論")
    matched_queries = candidate.get("matched_queries", [])
    if matched_queries:
        first_query = matched_queries[0].get("query")
        bits.append(f"由查詢「{first_query}」命中")
    if route_meta and route_meta.get("available"):
        bits.append(f"前後站直線路徑約 {route_meta.get('route_distance_km')} km")
    if not warnings:
        bits.append("目前沒有明顯警示")
    return "；".join(bits)


def _structured_constraint_check(
    slot: Dict[str, Any],
    candidate: Dict[str, Any],
) -> Tuple[float, List[str], List[str]]:
    """
    檢查候選點是否符合 slot 的結構化約束。
    回傳 (soft_penalty, warnings, failed_constraints)
    """
    constraints = slot.get("structured_constraints") or slot.get("confirmed_constraints") or {}
    if not constraints:
        return 0.0, [], []

    soft_penalty = 0.0
    warnings: List[str] = []
    failed_constraints: List[str] = []

    # min_rating
    min_rating = constraints.get("min_rating")
    if min_rating is not None:
        candidate_rating = candidate.get("rating")
        if candidate_rating is not None:
            if float(candidate_rating) < float(min_rating):
                soft_penalty += 0.15
                failed_constraints.append("min_rating")
        else:
            warnings.append("無法驗證評分門檻（缺評分資料）")

    # max_price_level
    max_price_level = constraints.get("max_price_level")
    if max_price_level is not None:
        candidate_price_level = candidate.get("price_level")
        if candidate_price_level is not None:
            if int(candidate_price_level) > int(max_price_level):
                soft_penalty += 0.15
                failed_constraints.append("max_price_level")
        else:
            warnings.append("無法驗證價格等級（缺 price_level 資料）")

    # open_until
    open_until = constraints.get("open_until")
    if open_until is not None:
        warnings.append("無法驗證營業至門檻（缺資料）")

    # requires_air_conditioning
    requires_ac = constraints.get("requires_air_conditioning")
    if requires_ac is not None:
        warnings.append("無法驗證空調要求（缺資料）")

    # requires_parking
    requires_parking = constraints.get("requires_parking")
    if requires_parking is not None:
        warnings.append("無法驗證停車位要求（缺資料）")

    soft_penalty = min(soft_penalty, 0.6)
    return soft_penalty, warnings, failed_constraints


def score_candidate(
    slot: Dict[str, Any],
    candidate: Dict[str, Any],
    slot_anchor_context: Optional[Dict[str, Any]] = None,
    destination_hint: str = "",
) -> Dict[str, Any]:
    relevance, matched_terms = _relevance_score(slot, candidate)
    quality, quality_warnings = _quality_score(candidate)
    opening, opening_warnings, opening_meta = _opening_score(candidate, slot)
    type_score, type_warnings = _type_score(slot, candidate)
    google_rank = _google_rank_score(candidate)
    route, route_warnings, route_meta = _route_score(candidate, slot_anchor_context)
    penalty, penalty_warnings = _status_penalty(candidate)
    location_penalty, location_warnings = _location_penalty(slot, candidate, destination_hint)
    constraint_penalty, constraint_warnings, failed_constraints = _structured_constraint_check(slot, candidate)
    total_penalty = _clamp(penalty + location_penalty + constraint_penalty)

    weighted = (
        WEIGHTS["relevance"] * relevance
        + WEIGHTS["quality"] * quality
        + WEIGHTS["opening"] * opening
        + WEIGHTS["route"] * route
        + WEIGHTS["google_rank"] * google_rank
        + WEIGHTS["type"] * type_score
    )
    relevance_gate = 0.55 + 0.45 * relevance
    total_score = _clamp((weighted - total_penalty) * relevance_gate)
    warnings = (
        quality_warnings
        + opening_warnings
        + route_warnings
        + type_warnings
        + penalty_warnings
        + location_warnings
        + constraint_warnings
    )
    eligible_for_llm = total_score >= 0.35 and total_penalty < 0.4

    return {
        "place_id": candidate.get("place_id"),
        "name": candidate.get("name"),
        "address": candidate.get("address"),
        "latitude": candidate.get("latitude"),
        "longitude": candidate.get("longitude"),
        "google_maps_uri": candidate.get("google_maps_uri"),
        "primary_type": candidate.get("primary_type"),
        "types": candidate.get("types", []),
        "rating": candidate.get("rating"),
        "user_rating_count": candidate.get("user_rating_count"),
        "business_status": candidate.get("business_status"),
        "matched_queries": candidate.get("matched_queries", []),
        "scores": {
            "total": round(total_score, 4),
            "relevance": round(relevance, 4),
            "quality": round(quality, 4),
            "opening": round(opening, 4),
            "route": round(route, 4),
            "google_rank": round(google_rank, 4),
            "type": round(type_score, 4),
            "status_penalty": round(penalty, 4),
            "location_penalty": round(location_penalty, 4),
            "constraint_penalty": round(constraint_penalty, 4),
            "penalty": round(total_penalty, 4),
            "relevance_gate": round(relevance_gate, 4),
        },
        "score_weights": WEIGHTS,
        "matched_terms": matched_terms,
        "warnings": warnings,
        "failed_constraints": failed_constraints,
        "eligible_for_llm": eligible_for_llm,
        "opening_meta": opening_meta,
        "route_meta": route_meta,
        "why": _reason_text(candidate, matched_terms, warnings, route_meta=route_meta),
        "raw_candidate": candidate,
    }


def check_candidates_against_constraints(
    candidates: List[Dict],
    constraints: Dict,
) -> Dict:
    """
    使用 z3 求解器檢查候選集合是否整體可滿足約束（min_rating, max_price_level）。
    若缺 z3，回傳友善訊息。
    """
    try:
        import z3
    except ImportError:
        return {
            "satisfiable": None,
            "satisfying_candidates": [],
            "diagnosis": "缺少 z3-solver 套件，請執行 pip install z3-solver",
        }

    min_rating = constraints.get("min_rating")
    max_price_level = constraints.get("max_price_level")

    solver = z3.Solver()
    bool_vars = []
    for idx, cand in enumerate(candidates):
        var = z3.Bool(f"cand_{idx}")
        bool_vars.append(var)
        if min_rating is not None:
            rating = cand.get("rating")
            if rating is not None:
                solver.add(
                    z3.Implies(
                        var,
                        z3.RealVal(float(rating)) >= z3.RealVal(float(min_rating)),
                    )
                )
        if max_price_level is not None:
            price_level = cand.get("price_level")
            if price_level is not None:
                solver.add(
                    z3.Implies(
                        var,
                        z3.IntVal(int(price_level)) <= z3.IntVal(int(max_price_level)),
                    )
                )
    solver.add(z3.Or(bool_vars))

    result = solver.check()
    satisfying_candidates: List = []
    diagnosis: str | None = None

    if result == z3.sat:
        model = solver.model()
        for var in bool_vars:
            if z3.is_true(model[var]):
                idx = int(str(var).split("_")[1])
                satisfying_candidates.append(candidates[idx].get("place_id"))
    elif result == z3.unsat:
        # 嘗試放寬其中一項約束
        constraints_to_try = {"min_rating", "max_price_level"}
        found_reason = False
        for removal in constraints_to_try:
            test_solver = z3.Solver()
            test_vars = []
            for idx, cand in enumerate(candidates):
                var = z3.Bool(f"cand_{idx}")
                test_vars.append(var)
                if min_rating is not None and removal != "min_rating":
                    rating = cand.get("rating")
                    if rating is not None:
                        test_solver.add(
                            z3.Implies(
                                var,
                                z3.RealVal(float(rating)) >= z3.RealVal(float(min_rating)),
                            )
                        )
                if max_price_level is not None and removal != "max_price_level":
                    price_level = cand.get("price_level")
                    if price_level is not None:
                        test_solver.add(
                            z3.Implies(
                                var,
                                z3.IntVal(int(price_level)) <= z3.IntVal(int(max_price_level)),
                            )
                        )
            test_solver.add(z3.Or(test_vars))
            if test_solver.check() == z3.sat:
                if removal == "min_rating":
                    diagnosis = "拿掉 min_rating 門檻後有解，建議放寬評分要求"
                else:
                    diagnosis = "拿掉 max_price_level 門檻後有解，建議放寬價格要求"
                found_reason = True
                break
        if not found_reason:
            diagnosis = "約束組合衝突，建議重新檢視需求"
    else:
        diagnosis = "求解器逾時或未知狀態"

    return {
        "satisfiable": result == z3.sat,
        "satisfying_candidates": satisfying_candidates,
        "diagnosis": diagnosis,
    }


def _compact_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "place_id": candidate.get("place_id"),
        "name": candidate.get("name"),
        "address": candidate.get("address"),
        "latitude": candidate.get("latitude"),
        "longitude": candidate.get("longitude"),
        "google_maps_uri": candidate.get("google_maps_uri"),
        "rating": candidate.get("rating"),
        "user_rating_count": candidate.get("user_rating_count"),
        "score": candidate.get("scores", {}).get("total"),
        "score_breakdown": candidate.get("scores"),
        "opening": candidate.get("opening_meta"),
        "route": candidate.get("route_meta"),
        "matched_queries": [
            item.get("query") for item in candidate.get("matched_queries", []) if item.get("query")
        ],
        "why": candidate.get("why"),
        "warnings": candidate.get("warnings", []),
        "selection_score": candidate.get("selection_score"),
    }


def _candidate_priority_terms(slot: Dict[str, Any], candidate: Dict[str, Any]) -> List[str]:
    profile = _slot_profile(slot)
    priority_terms = profile["priority_terms"]
    semantic_haystack = _candidate_semantic_text(candidate)
    return _matched_terms(priority_terms, semantic_haystack)


def _select_for_llm(
    slot: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    top_n: int,
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    covered_terms = set()
    remaining = list(candidates)

    while remaining and len(selected) < top_n:
        best_index = 0
        best_score = -1.0
        for index, candidate in enumerate(remaining):
            total = float(candidate.get("scores", {}).get("total") or 0.0)
            candidate_terms = set(_candidate_priority_terms(slot, candidate))
            new_term_count = len(candidate_terms - covered_terms)
            diversity_bonus = min(0.24, new_term_count * 0.16)
            selection_score = total + diversity_bonus
            if selection_score > best_score:
                best_score = selection_score
                best_index = index

        chosen = remaining.pop(best_index)
        chosen["selection_score"] = round(best_score, 4)
        selected.append(chosen)
        covered_terms.update(_candidate_priority_terms(slot, chosen))

    return selected


def _anchor_context_by_slot(anchor_context: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    if not isinstance(anchor_context, dict):
        return {}

    slots = anchor_context.get("slot_anchors") or anchor_context.get("candidate_slot_anchors") or []
    by_key: Dict[str, Dict[str, Any]] = {}
    for slot_anchor in slots:
        if not isinstance(slot_anchor, dict):
            continue
        key = slot_anchor.get("slot_key") or _slot_key(
            slot_anchor.get("day"),
            slot_anchor.get("time"),
            slot_anchor.get("original_activity"),
            slot_id=slot_anchor.get("slot_id"),
        )
        by_key[str(key)] = slot_anchor
    return by_key


def score_candidate_slots(
    places_result: Dict[str, Any],
    top_n: int = DEFAULT_TOP_N,
    run_id: Optional[str] = None,
    anchor_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    scored_slots: List[Dict[str, Any]] = []
    llm_slots: List[Dict[str, Any]] = []
    archived_records: List[Dict[str, Any]] = []
    anchors_by_slot = _anchor_context_by_slot(anchor_context)
    trip_context = places_result.get("trip_context")
    trip_context = trip_context if isinstance(trip_context, dict) else {}
    destination_hint = _compact_spaces(
        places_result.get("destination_hint")
        or trip_context.get("destination")
        or trip_context.get("city")
    )

    for slot_index, slot in enumerate(places_result.get("candidate_slots", []), start=1):
        slot_anchor_context = anchors_by_slot.get(
            _slot_key(
                slot.get("day"),
                slot.get("time"),
                slot.get("original_activity"),
                slot_id=slot.get("slot_id"),
            )
        )
        scored_candidates = [
            score_candidate(
                slot,
                candidate,
                slot_anchor_context=slot_anchor_context,
                destination_hint=destination_hint,
            )
            for candidate in slot.get("candidates", [])
        ]
        raw_candidates = slot.get("candidates", [])
        constraints = slot.get("structured_constraints") or slot.get("confirmed_constraints") or {}
        constraint_diagnosis = check_candidates_against_constraints(raw_candidates, constraints)

        scored_candidates.sort(
            key=lambda item: (
                item.get("scores", {}).get("total", 0),
            ),
            reverse=True,
        )

        selected = _select_for_llm(slot, scored_candidates, top_n)
        selected_ids = {item.get("place_id") for item in selected}
        archived = [
            item for item in scored_candidates if item.get("place_id") not in selected_ids
        ]

        for item in archived:
            archived_records.append(
                {
                    "run_id": run_id,
                    "archived_at": datetime.now().isoformat(timespec="seconds"),
                    "slot_id": slot.get("slot_id"),
                    "day": slot.get("day"),
                    "day_of_week": slot.get("day_of_week"),
                    "start_time": slot.get("start_time"),
                    "end_time": slot.get("end_time"),
                    "time": slot.get("time"),
                    "slot_type": slot.get("slot_type"),
                    "original_activity": slot.get("original_activity"),
                    "place_id": item.get("place_id"),
                    "name": item.get("name"),
                    "address": item.get("address"),
                    "score": item.get("scores", {}).get("total"),
                    "warnings": item.get("warnings", []),
                    "record": item,
                }
            )

        scored_slots.append(
            {
                "slot_index": slot_index,
                "slot_id": slot.get("slot_id"),
                "day": slot.get("day"),
                "day_of_week": slot.get("day_of_week"),
                "start_time": slot.get("start_time"),
                "end_time": slot.get("end_time"),
                "time": slot.get("time"),
                "slot_type": slot.get("slot_type"),
                "original_activity": slot.get("original_activity"),
                "reason_for_search": slot.get("reason_for_search"),
                "target_terms": slot.get("target_terms", []),
                "location_terms": slot.get("location_terms", []),
                "intent_type": slot.get("intent_type"),
                "must_have": slot.get("must_have", []),
                "queries": slot.get("queries", []),
                "candidate_count": len(scored_candidates),
                "selected_for_llm_count": len(selected),
                "archived_count": len(archived),
                "anchor_context": slot_anchor_context,
                "selected_for_llm": [_compact_candidate(item) for item in selected],
                "all_scored_candidates": scored_candidates,
                "constraint_diagnosis": constraint_diagnosis,
            }
        )

        llm_slots.append(
            {
                "slot_index": slot_index,
                "slot_id": slot.get("slot_id"),
                "day": slot.get("day"),
                "day_of_week": slot.get("day_of_week"),
                "start_time": slot.get("start_time"),
                "end_time": slot.get("end_time"),
                "time": slot.get("time"),
                "slot_type": slot.get("slot_type"),
                "original_activity": slot.get("original_activity"),
                "reason_for_search": slot.get("reason_for_search"),
                "target_terms": slot.get("target_terms", []),
                "location_terms": slot.get("location_terms", []),
                "intent_type": slot.get("intent_type"),
                "must_have": slot.get("must_have", []),
                "anchor_context": slot_anchor_context,
                "top_candidates": [_compact_candidate(item) for item in selected],
            }
        )

    return {
        "scoring_version": "candidate_scorer_v0.2",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        "top_n": top_n,
        "weights": WEIGHTS,
        "anchor_context_version": None
        if not isinstance(anchor_context, dict)
        else anchor_context.get("anchor_context_version"),
        "slots": scored_slots,
        "llm_input": {
            "instruction": "Only use these top candidates when replacing fuzzy itinerary slots. Do not invent new places.",
            "candidate_slots": llm_slots,
        },
        "archived_records": archived_records,
    }


def _archive_key(record: Dict[str, Any]) -> str:
    return "|".join(
        str(record.get(key) or "")
        for key in ("run_id", "slot_id", "day", "time", "original_activity", "place_id")
    )


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def append_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_keys = set()
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    existing_keys.add(_archive_key(json.loads(line)))
                except json.JSONDecodeError:
                    continue

    count = 0
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            key = _archive_key(record)
            if key in existing_keys:
                continue
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            existing_keys.add(key)
            count += 1
    return count


def save_scoring_outputs(
    scoring_result: Dict[str, Any],
    output_dir: Path,
    archive_path: Path = DEFAULT_ARCHIVE_PATH,
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    full_path = output_dir / "05_candidate_score_output.json"
    shortlist_path = output_dir / "05_candidate_shortlist_for_refinement.json"
    per_run_archive_path = output_dir / "05_candidate_archive.jsonl"

    full_path.write_text(json.dumps(scoring_result, ensure_ascii=False, indent=2), encoding="utf-8")
    shortlist_path.write_text(
        json.dumps(scoring_result["llm_input"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    archived_records = scoring_result.get("archived_records", [])
    per_run_count = write_jsonl(per_run_archive_path, archived_records)
    global_count = append_jsonl(archive_path, archived_records)

    return {
        "full_scoring": str(full_path),
        "shortlist_for_llm": str(shortlist_path),
        "per_run_archive": str(per_run_archive_path),
        "global_archive": str(archive_path),
        "per_run_archive_count": str(per_run_count),
        "global_archive_count": str(global_count),
    }


def _compact_spaces(value: Any) -> str:
    return " ".join(str(value or "").split())


def _planner_duration(planner_result: Dict[str, Any]) -> Tuple[int, int]:
    brief = planner_result.get("planning_brief_5w1h")
    when = brief.get("when") if isinstance(brief, dict) else {}
    days = when.get("duration_days") if isinstance(when, dict) else None
    nights = when.get("duration_nights") if isinstance(when, dict) else None

    if not days:
        itinerary = planner_result.get("itinerary")
        days = len(itinerary) if isinstance(itinerary, list) else 1
    if nights is None:
        nights = max(1, int(days) - 1)

    try:
        return max(1, int(days)), max(1, int(nights))
    except (TypeError, ValueError):
        return 1, 1


def _planning_budget(planner_result: Dict[str, Any]) -> Optional[float]:
    brief = planner_result.get("planning_brief_5w1h")
    how = brief.get("how") if isinstance(brief, dict) else {}
    if not isinstance(how, dict):
        return None
    try:
        budget = float(how.get("budget"))
    except (TypeError, ValueError):
        return None
    return budget if budget > 0 else None


def _price_value(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        for key in ("total", "per_night", "extracted_lowest", "lowest"):
            parsed = _price_value(value.get(key))
            if parsed is not None:
                return parsed
    text = str(value)
    digits = re.sub(r"[^\d.]", "", text)
    if not digits:
        return None
    try:
        return float(digits)
    except ValueError:
        return None


def _hotel_price(hotel: Dict[str, Any]) -> Dict[str, Any]:
    price = hotel.get("price") if isinstance(hotel.get("price"), dict) else {}
    per_night = _price_value(price.get("per_night"))
    total = _price_value(price.get("total"))
    nights = price.get("nights")
    try:
        nights = max(1, int(nights))
    except (TypeError, ValueError):
        nights = 1

    if per_night is None:
        per_night = _price_value(price.get("raw_rate_per_night"))
    if total is None:
        total = _price_value(price.get("raw_total_rate"))
    if per_night is None and total is not None:
        per_night = total / nights
    if total is None and per_night is not None:
        total = per_night * nights

    return {
        "currency": price.get("currency"),
        "per_night": None if per_night is None else round(per_night, 2),
        "total": None if total is None else round(total, 2),
        "nights": nights,
        "availability_confirmed": price.get("availability_confirmed"),
        "source": price.get("source"),
    }


def _review_score(candidate: Dict[str, Any]) -> float:
    try:
        count = float(candidate.get("user_rating_count") or 0)
    except (TypeError, ValueError):
        count = 0.0
    if count <= 0:
        return 0.35
    return _clamp(math.log10(count + 1) / math.log10(1000 + 1))


def _rating_score(candidate: Dict[str, Any]) -> float:
    try:
        rating = float(candidate.get("rating"))
    except (TypeError, ValueError):
        return 0.55
    return _clamp(rating / 5.0)


def _rank_score(candidate: Dict[str, Any]) -> float:
    signals = candidate.get("ranking_signals") if isinstance(candidate.get("ranking_signals"), dict) else {}
    try:
        return _clamp(float(signals.get("google_rank_score")))
    except (TypeError, ValueError):
        return 0.4


def _distance_summary(
    candidate: Dict[str, Any],
    reference_points: List[Dict[str, Any]],
) -> Dict[str, Any]:
    candidate_coordinates = _coordinates(candidate)
    distances = []
    if candidate_coordinates:
        for point in reference_points:
            point_coordinates = _coordinates(point)
            if point_coordinates:
                distances.append(_haversine_km(candidate_coordinates, point_coordinates))

    if not distances:
        return {
            "average_km": None,
            "nearest_km": None,
            "score": 0.55,
            "reference_count": len(reference_points),
        }

    average_km = sum(distances) / len(distances)
    nearest_km = min(distances)
    score = 0.70 * math.exp(-average_km / 12.0) + 0.30 * math.exp(-nearest_km / 4.0)
    return {
        "average_km": round(average_km, 3),
        "nearest_km": round(nearest_km, 3),
        "score": round(_clamp(score), 4),
        "reference_count": len(distances),
    }


def _price_score(price: Dict[str, Any], planner_result: Dict[str, Any], lodging_result: Dict[str, Any]) -> float:
    total = price.get("total")
    per_night = price.get("per_night")
    if total is None and per_night is None:
        return 0.45

    search_context = lodging_result.get("search_context")
    search_context = search_context if isinstance(search_context, dict) else {}
    cap = _price_value(search_context.get("budget_per_night_cap"))
    budget = _planning_budget(planner_result)
    _, nights = _planner_duration(planner_result)
    if cap is None and budget:
        cap = budget / max(1, nights)

    if cap is None or cap <= 0 or per_night is None:
        return 0.60

    ratio = float(per_night) / cap
    if ratio <= 1.0:
        return 1.0
    if ratio <= 1.5:
        return 0.72
    if ratio <= 2.5:
        return 0.45
    return 0.18


def score_lodging_candidate(
    hotel: Dict[str, Any],
    planner_result: Dict[str, Any],
    lodging_result: Dict[str, Any],
    reference_points: List[Dict[str, Any]],
) -> Dict[str, Any]:
    price = _hotel_price(hotel)
    distance = _distance_summary(hotel, reference_points)
    scores = {
        "rating": round(_rating_score(hotel), 4),
        "reviews": round(_review_score(hotel), 4),
        "google_rank": round(_rank_score(hotel), 4),
        "price": round(_price_score(price, planner_result, lodging_result), 4),
        "distance": distance["score"],
    }
    total = (
        scores["rating"] * 0.18
        + scores["reviews"] * 0.12
        + scores["google_rank"] * 0.16
        + scores["price"] * 0.24
        + scores["distance"] * 0.30
    )
    warnings = []
    if price.get("per_night") is None:
        warnings.append("缺少每晚價格")
    if distance.get("average_km") is None:
        warnings.append("缺少可比較的景點座標，住宿距離只給中性分")
    if hotel.get("rating") is None:
        warnings.append("缺少住宿評分")

    return {
        "candidate_domain": "lodging",
        "place_id": hotel.get("place_id") or hotel.get("property_token") or hotel.get("name"),
        "property_token": hotel.get("property_token"),
        "name": hotel.get("name"),
        "address": hotel.get("address"),
        "latitude": hotel.get("latitude"),
        "longitude": hotel.get("longitude"),
        "rating": hotel.get("rating"),
        "user_rating_count": hotel.get("user_rating_count"),
        "hotel_class": hotel.get("hotel_class"),
        "amenities": hotel.get("amenities", []),
        "price": price,
        "matched_queries": hotel.get("matched_queries", []),
        "scores": {**scores, "total": round(total, 4)},
        "distance_to_place_candidates": distance,
        "warnings": warnings,
        "raw_candidate": hotel,
    }


def _compact_lodging_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "candidate_domain": "lodging",
        "place_id": candidate.get("place_id"),
        "property_token": candidate.get("property_token"),
        "name": candidate.get("name"),
        "address": candidate.get("address"),
        "latitude": candidate.get("latitude"),
        "longitude": candidate.get("longitude"),
        "rating": candidate.get("rating"),
        "user_rating_count": candidate.get("user_rating_count"),
        "hotel_class": candidate.get("hotel_class"),
        "price": candidate.get("price"),
        "scores": candidate.get("scores"),
        "distance_to_place_candidates": candidate.get("distance_to_place_candidates"),
        "warnings": candidate.get("warnings", []),
    }


def _selected_place_points(place_scoring_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    points: List[Dict[str, Any]] = []
    for slot in place_scoring_result.get("slots", []):
        for candidate in slot.get("selected_for_llm", []):
            if _coordinates(candidate):
                item = dict(candidate)
                item["candidate_domain"] = "place"
                item["slot_id"] = slot.get("slot_id")
                item["day"] = slot.get("day")
                item["original_activity"] = slot.get("original_activity")
                points.append(item)
    return points


def _geo_point(candidate: Dict[str, Any], domain: str) -> Optional[Dict[str, Any]]:
    coordinates = _coordinates(candidate)
    if coordinates is None:
        return None
    latitude, longitude = coordinates
    return {
        "candidate_domain": domain,
        "place_id": candidate.get("place_id"),
        "name": candidate.get("name"),
        "latitude": latitude,
        "longitude": longitude,
        "score": candidate.get("scores", {}).get("total") or candidate.get("score"),
        "slot_id": candidate.get("slot_id"),
        "day": candidate.get("day"),
        "original_activity": candidate.get("original_activity"),
    }


def _suggest_cluster_k(points: List[Dict[str, Any]], planner_result: Dict[str, Any]) -> int:
    if not points:
        return 0
    days, _ = _planner_duration(planner_result)
    if len(points) <= 2:
        return 1
    return max(1, min(3, days - 1, len(points)))


def _mean_center(points: List[Dict[str, Any]]) -> Tuple[float, float]:
    return (
        sum(float(point["latitude"]) for point in points) / len(points),
        sum(float(point["longitude"]) for point in points) / len(points),
    )


def _kmeans_geo(points: List[Dict[str, Any]], k: Optional[int] = None, iterations: int = 12) -> Dict[str, Any]:
    usable_points = [point for point in points if _coordinates(point)]
    if not usable_points:
        return {"k": 0, "clusters": []}

    safe_k = k or min(2, len(usable_points))
    safe_k = max(1, min(int(safe_k), len(usable_points)))
    sorted_points = sorted(
        usable_points,
        key=lambda item: (
            float(item.get("longitude") or 0.0),
            float(item.get("latitude") or 0.0),
            str(item.get("name") or ""),
        ),
    )
    if safe_k == 1:
        seeds = [sorted_points[len(sorted_points) // 2]]
    else:
        seeds = [
            sorted_points[round(index * (len(sorted_points) - 1) / (safe_k - 1))]
            for index in range(safe_k)
        ]
    centers = [(float(seed["latitude"]), float(seed["longitude"])) for seed in seeds]

    assignments = [0 for _ in usable_points]
    for _ in range(iterations):
        changed = False
        for index, point in enumerate(usable_points):
            point_coordinates = (float(point["latitude"]), float(point["longitude"]))
            nearest_index = min(
                range(len(centers)),
                key=lambda center_index: _haversine_km(point_coordinates, centers[center_index]),
            )
            if assignments[index] != nearest_index:
                assignments[index] = nearest_index
                changed = True

        if not changed:
            break

        for center_index in range(safe_k):
            members = [
                point for point, assignment in zip(usable_points, assignments) if assignment == center_index
            ]
            if members:
                centers[center_index] = _mean_center(members)

    clusters = []
    for center_index, center in enumerate(centers):
        members = [
            point for point, assignment in zip(usable_points, assignments) if assignment == center_index
        ]
        clusters.append(
            {
                "cluster_id": center_index + 1,
                "center": {
                    "latitude": round(center[0], 7),
                    "longitude": round(center[1], 7),
                },
                "member_count": len(members),
                "members": members,
            }
        )
    clusters.sort(key=lambda cluster: cluster["member_count"], reverse=True)
    return {"k": safe_k, "clusters": clusters}


def _lodging_strategy(planner_result: Dict[str, Any], clusters: Dict[str, Any]) -> Dict[str, Any]:
    days, nights = _planner_duration(planner_result)
    preferred_mode = "single_hotel" if nights <= 3 else "single_or_split_hotel"
    max_switches = 0 if nights <= 3 else min(2, max(0, nights - 1))
    return {
        "duration_days": days,
        "duration_nights": nights,
        "preferred_mode": preferred_mode,
        "max_hotel_switches": max_switches,
        "cluster_count": clusters.get("k", 0),
        "rule": "短天數預設盡量住同一間；只有地理分布很遠或跨城市時才換住宿。",
    }


def build_combined_candidate_package(
    planner_result: Dict[str, Any],
    places_result: Dict[str, Any],
    lodging_result: Dict[str, Any],
    place_top_n: int = 5,
    lodging_top_n: int = 8,
    cluster_k: Optional[int] = None,
    run_id: Optional[str] = None,
    anchor_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    place_scoring = score_candidate_slots(
        places_result,
        top_n=place_top_n,
        run_id=run_id,
        anchor_context=anchor_context,
    )
    place_points = _selected_place_points(place_scoring)

    lodging_candidates = [
        score_lodging_candidate(hotel, planner_result, lodging_result, place_points)
        for hotel in lodging_result.get("candidates", [])
        if isinstance(hotel, dict)
    ]
    lodging_candidates.sort(
        key=lambda item: item.get("scores", {}).get("total", 0.0),
        reverse=True,
    )
    selected_lodging = lodging_candidates[:lodging_top_n]

    geo_points: List[Dict[str, Any]] = []
    for point in place_points:
        geo_point = _geo_point(point, "place")
        if geo_point:
            geo_points.append(geo_point)
    for hotel in selected_lodging:
        geo_point = _geo_point(hotel, "lodging")
        if geo_point:
            geo_points.append(geo_point)

    resolved_cluster_k = cluster_k or _suggest_cluster_k(geo_points, planner_result)
    clusters = _kmeans_geo(geo_points, k=resolved_cluster_k)
    lodging_strategy = _lodging_strategy(planner_result, clusters)

    llm_input = {
        "instruction": (
            "Use only the provided place and lodging candidates. Decide whether each fuzzy slot can be "
            "replaced directly, needs minor itinerary reordering, or requires a larger itinerary rewrite. "
            "Also decide lodging for each night. Do not invent new places or hotels."
        ),
        "decision_schema": {
            "fuzzy_slot_decision": "single_replace | minor_adjustment | major_rebuild",
            "lodging_decision": "single_hotel | split_hotels",
        },
        "trip_context": planner_result.get("trip_context", {}),
        "planning_brief_5w1h": planner_result.get("planning_brief_5w1h", {}),
        "original_itinerary": planner_result.get("itinerary", []),
        "candidate_slots": place_scoring["llm_input"]["candidate_slots"],
        "lodging_candidates": [_compact_lodging_candidate(item) for item in selected_lodging],
        "geo_clusters": clusters,
        "lodging_strategy": lodging_strategy,
    }

    return {
        "combined_candidate_version": "candidate_combined_v0.1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        "source_versions": {
            "places_search_version": places_result.get("places_search_version"),
            "lodging_search_version": lodging_result.get("lodging_search_version"),
            "place_scoring_version": place_scoring.get("scoring_version"),
        },
        "place_scoring": place_scoring,
        "lodging_scoring": {
            "candidate_count": len(lodging_candidates),
            "selected_for_llm_count": len(selected_lodging),
            "selected_for_llm": [_compact_lodging_candidate(item) for item in selected_lodging],
            "all_scored_lodging_candidates": lodging_candidates,
        },
        "geo_clusters": clusters,
        "lodging_strategy": lodging_strategy,
        "llm_input": llm_input,
    }


def save_combined_candidate_outputs(
    combined_result: Dict[str, Any],
    output_dir: Path,
    full_filename: str,
    llm_filename: str,
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    full_path = output_dir / full_filename
    llm_path = output_dir / llm_filename
    full_path.write_text(json.dumps(combined_result, ensure_ascii=False, indent=2), encoding="utf-8")
    llm_path.write_text(
        json.dumps(combined_result["llm_input"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "combined_candidate": str(full_path),
        "llm_input": str(llm_path),
    }


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent

    input_path = (
        Path(sys.argv[1]).resolve()
        if len(sys.argv) > 1
        else base_dir / "traced_outputs" / "北京上海測資_result.json"
    )
    places_path = (
        Path(sys.argv[2]).resolve()
        if len(sys.argv) > 2
        else input_path.parent / f"{input_path.stem}_places_search.json"
    )
    lodging_path = (
        Path(sys.argv[3]).resolve()
        if len(sys.argv) > 3
        else input_path.parent / f"{input_path.stem}_hotel_info.json"
    )

    with open(input_path, "r", encoding="utf-8") as f:
        planner_context = json.load(f)
    with open(places_path, "r", encoding="utf-8") as f:
        places_context = json.load(f)
    with open(lodging_path, "r", encoding="utf-8") as f:
        lodging_context = json.load(f)

    PLACE_TOP_N = 5
    LODGING_TOP_N = 8
    KMEANS_K = None

    # print("=" * 20 + "看一下輸入檔" + "=" * 20)
    # print("planner:", input_path)
    # print("places:", places_path)
    # print("lodging:", lodging_path)
    # print("=" * 20 + "結束" + "=" * 20)
    # print("\n")

    combined_result = build_combined_candidate_package(
        planner_context,
        places_context,
        lodging_context,
        place_top_n=PLACE_TOP_N,
        lodging_top_n=LODGING_TOP_N,
        cluster_k=KMEANS_K,
        run_id=input_path.stem,
    )

    # print("=" * 20 + "看一下景點候選" + "=" * 20)
    # for slot in combined_result["llm_input"]["candidate_slots"]:
    #     print(slot.get("slot_id"), slot.get("original_activity"))
    #     for candidate in slot.get("top_candidates", []):
    #         print("-", candidate.get("name"), "score:", candidate.get("score"))
    #     print()
    # print("=" * 20 + "結束" + "=" * 20)
    # print("\n")

    # print("=" * 20 + "看一下住宿候選" + "=" * 20)
    # for hotel in combined_result["llm_input"]["lodging_candidates"]:
    #     price = hotel.get("price") or {}
    #     scores = hotel.get("scores") or {}
    #     print(
    #         "-",
    #         hotel.get("name"),
    #         "score:",
    #         scores.get("total"),
    #         "per_night:",
    #         price.get("per_night"),
    #         price.get("currency"),
    #     )
    # print("=" * 20 + "結束" + "=" * 20)
    # print("\n")

    print("=" * 20 + "看一下分群" + "=" * 20)
    print("k =", combined_result["geo_clusters"].get("k"))
    for cluster in combined_result["geo_clusters"].get("clusters", []):
        print("cluster", cluster.get("cluster_id"), "members:", cluster.get("member_count"))
        for member in cluster.get("members", [])[:8]:
            print("-", member.get("candidate_domain"), member.get("name"))
        print()
    print("=" * 20 + "結束" + "=" * 20)
    print("\n")

    filename = f"{input_path.stem}_candidate_package.json"
    llm_filename = f"{input_path.stem}_candidate_llm_input.json"
    output_path = base_dir / "traced_outputs"
    saved_path = save_combined_candidate_outputs(
        combined_result,
        output_path,
        filename,
        llm_filename,
    )

    # print("=" * 20 + "看一下保存位置" + "=" * 20)
    # print(saved_path)
    # print("=" * 20 + "結束" + "=" * 20)
