import argparse
from datetime import datetime
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PlacesSearchAgent import GooglePlacesTextSearch, _normalize_place


STORE_DIR = Path(__file__).resolve().parent / "candidate_store"
DEFAULT_ANCHOR_CACHE_PATH = STORE_DIR / "anchor_cache.json"
DEFAULT_START_ANCHOR = os.getenv("TRAVEL_DEFAULT_START_ANCHOR", "台南火車站")
DEFAULT_END_ANCHOR = os.getenv("TRAVEL_DEFAULT_END_ANCHOR", "台南火車站")

COMMUTE_TERMS = {"預留通勤時間", "通勤時間", "交通時間"}
GENERIC_ACTIVITY_TERMS = {
    "飯店",
    "早餐",
    "午餐",
    "晚餐",
    "小吃",
    "美食",
    "返程",
    "準備返程",
}
LEADING_PHRASES = [
    "搭乘高鐵抵達",
    "搭乘火車抵達",
    "搭乘台鐵抵達",
    "抵達",
    "參觀",
    "造訪",
    "探索",
    "夜遊",
    "休憩於",
    "前往",
    "漫步",
    "入住飯店並享用",
    "入住飯店",
    "返回飯店並準備",
    "返回飯店",
    "享用",
    "品嚐",
    "品嘗",
    "在",
]


def _read_json_from_path_or_stdin(path: Optional[str]) -> Dict[str, Any]:
    if path:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    raw = sys.stdin.read().strip()
    if not raw:
        raise SystemExit("Please pass a JSON path or pipe JSON through stdin.")
    return json.loads(raw)

#做資料標準化，全部變成dict
def _as_json_object(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, list):
        return {"pending_searches": raw}
    if isinstance(raw, str):
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return {"pending_searches": parsed}
        if isinstance(parsed, dict):
            return parsed
    raise TypeError("Expected dict, list, JSON object string, or JSON array string.")


def _extract_pending_searches(raw: Any) -> List[Dict[str, Any]]:
    obj = _as_json_object(raw)
    pending = obj.get("pending_searches", obj)
    if not isinstance(pending, list):
        raise ValueError("Expected {'pending_searches': [...]} or a JSON array.")
    return [item for item in pending if isinstance(item, dict)]


def _compact_spaces(text: Any) -> str:
    return " ".join(str(text or "").split())


def _normalize_text(text: Any) -> str:
    return _compact_spaces(text).replace("臺", "台").lower()


def _slot_key(day: Any, time: Any, activity: Any, slot_id: Any = None) -> str:
    if slot_id:
        return f"slot_id:{slot_id}"
    return "|".join([str(day or ""), str(time or ""), _normalize_text(activity)])


def _slot_key_for_item(item: Dict[str, Any], activity_key: str = "activity") -> str:
    return _slot_key(
        item.get("day"),
        item.get("time"),
        item.get(activity_key),
        slot_id=item.get("slot_id"),
    )


def _cache_key(query: str, destination_hint: str) -> str:
    return _normalize_text(f"{destination_hint}|{query}")


def _load_cache(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def _save_cache(path: Path, cache: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def _is_commute(item_or_activity: Any) -> bool:
    if isinstance(item_or_activity, dict):
        if item_or_activity.get("is_commute") is True:
            return True
        if str(item_or_activity.get("slot_type") or "").lower() == "transport":
            return True
        item_or_activity = item_or_activity.get("activity")
    text = str(item_or_activity or "")
    return any(term in text for term in COMMUTE_TERMS)


def _is_structured_searchable(item: Dict[str, Any]) -> bool:
    return item.get("is_searchable") is True


def _strip_parentheses(text: str) -> str:
    return re.sub(r"[（(].*?[）)]", "", text).strip()


def _strip_leading_phrases(text: str) -> str:
    current = text.strip()
    changed = True
    while changed:
        changed = False
        for phrase in LEADING_PHRASES:
            if current.startswith(phrase) and len(current) > len(phrase):
                current = current[len(phrase) :].strip()
                changed = True
    return current


def _split_activity_components(text: str) -> List[str]:
    parts = re.split(r"和|與|及|、|/|／", text)
    cleaned = []
    for part in parts:
        item = _strip_leading_phrases(_strip_parentheses(part)).strip(" ，,。")
        if item:
            cleaned.append(item)
    return cleaned or [text]


def _looks_generic_activity(text: str) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return True
    return normalized in GENERIC_ACTIVITY_TERMS


def _with_destination_hint(query: str, destination_hint: str) -> str:
    query = _compact_spaces(query)
    if destination_hint and query and destination_hint not in query:
        return f"{destination_hint} {query}"
    return query


def _query_for_activity(
    activity: Any,
    role: str,
    destination_hint: str,
    default_anchor: str,
    force_default: bool = False,
) -> Tuple[str, List[str]]:
    warnings: List[str] = []
    if force_default:
        return default_anchor, warnings

    text = _strip_leading_phrases(_strip_parentheses(str(activity or ""))).strip(" ，,。")
    components = _split_activity_components(text)
    chosen = components[-1] if role == "previous" else components[0]
    chosen = _strip_leading_phrases(chosen).strip(" ，,。")

    if _looks_generic_activity(chosen):
        warnings.append(f"anchor 活動「{activity}」太抽象，改用預設 anchor：{default_anchor}")
        return default_anchor, warnings

    return _with_destination_hint(chosen, destination_hint), warnings


def _query_for_anchor_item(
    item: Optional[Dict[str, Any]],
    role: str,
    destination_hint: str,
    default_anchor: str,
    force_default: bool = False,
) -> Tuple[str, List[str]]:
    warnings: List[str] = []
    if force_default or item is None:
        return default_anchor, warnings

    route_key = "route_exit_place" if role == "previous" else "route_entry_place"
    for key in (route_key, "place_query"):
        value = _compact_spaces(item.get(key))
        if value:
            return _with_destination_hint(value, destination_hint), warnings

    fallback_query, fallback_warnings = _query_for_activity(
        item.get("activity"),
        role=role,
        destination_hint=destination_hint,
        default_anchor=default_anchor,
    )
    fallback_warnings.append(
        "此 anchor 缺少 place_query/route_entry_place/route_exit_place，已改用文字清理 fallback。"
    )
    return fallback_query, warnings + fallback_warnings


def _day_number(day_obj: Dict[str, Any], fallback: int) -> Any:
    return day_obj.get("day", fallback)


def _schedule_for_day(day_obj: Dict[str, Any], fallback_day: int) -> List[Dict[str, Any]]:
    day = _day_number(day_obj, fallback_day)
    day_of_week = day_obj.get("day_of_week")
    schedule = day_obj.get("schedule", [])
    if not isinstance(schedule, list):
        return []

    items: List[Dict[str, Any]] = []
    for index, item in enumerate(schedule):
        if not isinstance(item, dict):
            continue
        items.append(
            {
                "slot_id": item.get("slot_id"),
                "day": day,
                "day_of_week": item.get("day_of_week", day_of_week),
                "start_time": item.get("start_time"),
                "end_time": item.get("end_time"),
                "time": item.get("time"),
                "activity": item.get("activity"),
                "slot_type": item.get("slot_type"),
                "is_commute": item.get("is_commute"),
                "is_searchable": item.get("is_searchable"),
                "place_query": item.get("place_query"),
                "route_entry_place": item.get("route_entry_place"),
                "route_exit_place": item.get("route_exit_place"),
                "locked": item.get("locked"),
                "schedule_index": index,
            }
        )
    return items


def _build_schedule_by_day(itinerary_json: Dict[str, Any]) -> Dict[Any, List[Dict[str, Any]]]:
    days = itinerary_json.get("itinerary", [])
    if not isinstance(days, list):
        return {}
    return {
        _day_number(day_obj, index + 1): _schedule_for_day(day_obj, index + 1)
        for index, day_obj in enumerate(days)
        if isinstance(day_obj, dict)
    }


def _get_day_schedule(
    schedule_by_day: Dict[Any, List[Dict[str, Any]]],
    day: Any,
) -> List[Dict[str, Any]]:
    if day in schedule_by_day:
        return schedule_by_day[day]
    for stored_day, schedule in schedule_by_day.items():
        if str(stored_day) == str(day):
            return schedule
    return []


def _pending_keys(pending_searches: List[Dict[str, Any]]) -> set:
    return {
        _slot_key(
            item.get("day"),
            item.get("time"),
            item.get("original_activity"),
            slot_id=item.get("slot_id"),
        )
        for item in pending_searches
    }


def _matches_pending_item(schedule_item: Dict[str, Any], pending: Dict[str, Any]) -> bool:
    schedule_slot_id = schedule_item.get("slot_id")
    pending_slot_id = pending.get("slot_id")
    if schedule_slot_id and pending_slot_id:
        return str(schedule_slot_id) == str(pending_slot_id)

    if str(schedule_item.get("day")) != str(pending.get("day")):
        return False
    if str(schedule_item.get("time")) != str(pending.get("time")):
        return False

    schedule_activity = _normalize_text(schedule_item.get("activity"))
    pending_activity = _normalize_text(pending.get("original_activity"))
    return (
        schedule_activity == pending_activity
        or schedule_activity in pending_activity
        or pending_activity in schedule_activity
    )


def _find_slot_position(
    schedule_by_day: Dict[Any, List[Dict[str, Any]]],
    pending: Dict[str, Any],
) -> Tuple[Optional[Any], Optional[int]]:
    pending_day = pending.get("day")
    day_schedule = _get_day_schedule(schedule_by_day, pending_day)
    for index, item in enumerate(day_schedule):
        if _matches_pending_item(item, pending):
            return item.get("day", pending_day), index
    return None, None


def _is_searchable_slot(
    item: Dict[str, Any],
    searchable_keys: set,
) -> bool:
    return _is_structured_searchable(item) or _slot_key_for_item(item) in searchable_keys


def _nearest_anchor_item(
    day_schedule: List[Dict[str, Any]],
    slot_index: int,
    direction: int,
    searchable_keys: set,
) -> Optional[Dict[str, Any]]:
    index = slot_index + direction
    while 0 <= index < len(day_schedule):
        item = day_schedule[index]
        if not _is_commute(item) and not _is_searchable_slot(item, searchable_keys):
            return item
        index += direction
    return None


def _resolve_place(
    query: str,
    destination_hint: str,
    cache: Dict[str, Any],
    searcher: GooglePlacesTextSearch,
) -> Dict[str, Any]:
    key = _cache_key(query, destination_hint)
    if key in cache:
        cached = dict(cache[key])
        cached["cache_hit"] = True
        return cached

    try:
        places = searcher.search_text(query, page_size=1)
        place = _normalize_place(places[0], query, "anchor", 0) if places else None
        entry = {
            "query": query,
            "resolved_at": datetime.now().isoformat(timespec="seconds"),
            "place": place,
            "error": None if place else "Google Places returned no anchor candidate.",
            "cache_hit": False,
        }
    except Exception as exc:
        entry = {
            "query": query,
            "resolved_at": datetime.now().isoformat(timespec="seconds"),
            "place": None,
            "error": str(exc),
            "cache_hit": False,
        }

    if entry.get("place") is not None:
        cache[key] = {key_item: value for key_item, value in entry.items() if key_item != "cache_hit"}
    return entry


def _make_anchor_ref(
    item: Optional[Dict[str, Any]],
    role: str,
    destination_hint: str,
    default_anchor: str,
    source: str,
    cache: Dict[str, Any],
    searcher: GooglePlacesTextSearch,
    force_default: bool = False,
) -> Dict[str, Any]:
    activity = default_anchor if item is None else item.get("activity")
    query, warnings = _query_for_anchor_item(
        item,
        role=role,
        destination_hint=destination_hint,
        default_anchor=default_anchor,
        force_default=force_default or item is None,
    )
    resolved = _resolve_place(query, destination_hint, cache, searcher)
    if resolved.get("error"):
        warnings.append(str(resolved["error"]))

    return {
        "source": source,
        "role": role,
        "slot_id": None if item is None else item.get("slot_id"),
        "day": None if item is None else item.get("day"),
        "day_of_week": None if item is None else item.get("day_of_week"),
        "start_time": None if item is None else item.get("start_time"),
        "end_time": None if item is None else item.get("end_time"),
        "time": None if item is None else item.get("time"),
        "activity": activity,
        "query": query,
        "place": resolved.get("place"),
        "cache_hit": resolved.get("cache_hit", False),
        "warnings": warnings,
    }


def _trip_context(itinerary_json: Dict[str, Any]) -> Dict[str, Any]:
    context = itinerary_json.get("trip_context")
    return context if isinstance(context, dict) else {}


def _anchor_query_from_context(anchor: Any) -> Optional[str]:
    if isinstance(anchor, dict):
        for key in ("place_query", "name", "address"):
            value = _compact_spaces(anchor.get(key))
            if value:
                return value
    if isinstance(anchor, str) and anchor.strip():
        return _compact_spaces(anchor)
    return None


def _context_destination(context: Dict[str, Any]) -> str:
    for key in ("destination", "city", "area"):
        value = _compact_spaces(context.get(key))
        if value:
            return value
    return ""


def _context_anchor_query(
    context: Dict[str, Any],
    keys: Tuple[str, ...],
    explicit_value: Optional[str],
    fallback: str,
) -> str:
    if explicit_value:
        return explicit_value
    for key in keys:
        query = _anchor_query_from_context(context.get(key))
        if query:
            return query
    return fallback


def resolve_anchor_context(
    itinerary_json: Dict[str, Any],
    pending_searches_json: Any,
    destination_hint: str = "",
    start_anchor: Optional[str] = None,
    end_anchor: Optional[str] = None,
    cache_path: Path = DEFAULT_ANCHOR_CACHE_PATH,
) -> Dict[str, Any]:
    pending_searches = _extract_pending_searches(pending_searches_json)
    schedule_by_day = _build_schedule_by_day(itinerary_json)
    searchable_keys = _pending_keys(pending_searches)
    cache = _load_cache(cache_path)
    searcher = GooglePlacesTextSearch(include_quality_fields=False)
    slot_anchors: List[Dict[str, Any]] = []
    context = _trip_context(itinerary_json)
    resolved_destination_hint = destination_hint or _context_destination(context)
    resolved_start_anchor = _context_anchor_query(
        context,
        ("arrival_anchor", "start_anchor", "origin_anchor"),
        start_anchor,
        DEFAULT_START_ANCHOR,
    )
    resolved_end_anchor = _context_anchor_query(
        context,
        ("return_anchor", "end_anchor", "departure_anchor"),
        end_anchor,
        DEFAULT_END_ANCHOR,
    )

    day_order = list(schedule_by_day.keys())
    first_day = day_order[0] if day_order else None
    last_day = day_order[-1] if day_order else None

    for pending in pending_searches:
        day, slot_index = _find_slot_position(schedule_by_day, pending)
        day_schedule = _get_day_schedule(schedule_by_day, day) if day is not None else []
        warnings: List[str] = []

        if day is None or slot_index is None:
            warnings.append("找不到模糊 slot 在 planner 行程中的位置，前後 anchor 改用 trip_context/default anchor。")
            previous_anchor = _make_anchor_ref(
                None,
                "previous",
                resolved_destination_hint,
                resolved_start_anchor,
                "default_start",
                cache,
                searcher,
                force_default=True,
            )
            next_anchor = _make_anchor_ref(
                None,
                "next",
                resolved_destination_hint,
                resolved_end_anchor,
                "default_end",
                cache,
                searcher,
                force_default=True,
            )
        else:
            previous_item = _nearest_anchor_item(day_schedule, slot_index, -1, searchable_keys)
            next_item = _nearest_anchor_item(day_schedule, slot_index, 1, searchable_keys)

            previous_is_start_boundary = (
                previous_item is not None
                and str(day) == str(first_day)
                and previous_item.get("schedule_index") == 0
            )
            next_is_end_boundary = (
                next_item is not None
                and str(day) == str(last_day)
                and next_item.get("schedule_index") == len(day_schedule) - 1
            )

            previous_anchor = _make_anchor_ref(
                previous_item,
                "previous",
                resolved_destination_hint,
                resolved_start_anchor,
                "default_start" if previous_item is None or previous_is_start_boundary else "schedule",
                cache,
                searcher,
                force_default=previous_item is None or previous_is_start_boundary,
            )
            next_anchor = _make_anchor_ref(
                next_item,
                "next",
                resolved_destination_hint,
                resolved_end_anchor,
                "default_end" if next_item is None or next_is_end_boundary else "schedule",
                cache,
                searcher,
                force_default=next_item is None or next_is_end_boundary,
            )

        slot_anchors.append(
            {
                "slot_key": _slot_key(
                    pending.get("day"),
                    pending.get("time"),
                    pending.get("original_activity"),
                    slot_id=pending.get("slot_id"),
                ),
                "slot_id": pending.get("slot_id"),
                "day": pending.get("day"),
                "day_of_week": pending.get("day_of_week"),
                "start_time": pending.get("start_time"),
                "end_time": pending.get("end_time"),
                "time": pending.get("time"),
                "slot_type": pending.get("slot_type"),
                "original_activity": pending.get("original_activity"),
                "previous_anchor": previous_anchor,
                "next_anchor": next_anchor,
                "warnings": warnings
                + previous_anchor.get("warnings", [])
                + next_anchor.get("warnings", []),
            }
        )

    _save_cache(cache_path, cache)
    return {
        "anchor_context_version": "anchor_resolver_v0.1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "trip_context": context,
        "destination_hint": resolved_destination_hint,
        "default_start_anchor": resolved_start_anchor,
        "default_end_anchor": resolved_end_anchor,
        "distance_note": "Anchor Resolver only resolves nearby schedule anchors. CandidateScorer uses straight-line distance as a routing proxy.",
        "slot_anchors": slot_anchors,
    }

#作廢，整個改完記得刪，不夠通用
def save_anchor_context_output(
    anchor_context: Dict[str, Any],
    output_dir: Path,
    filename: str = "03_anchor_context_output.json",
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    path.write_text(json.dumps(anchor_context, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"anchor_context": str(path)}

def save_json_output(data: Dict[str, Any], output_dir: Path, filename: str, result_key: str) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return {result_key: str(path)}

def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve previous/next itinerary anchors for fuzzy slots.")
    parser.add_argument("itinerary", help="Path to 01_planner_agent_output.json")
    parser.add_argument("pending_searches", help="Path to 02_extract_searchable_activities_output.json")
    parser.add_argument("--destination-hint", default="")
    parser.add_argument("--start-anchor")
    parser.add_argument("--end-anchor")
    parser.add_argument("--cache-path", default=str(DEFAULT_ANCHOR_CACHE_PATH))
    parser.add_argument("--output-dir", help="Directory for 03_anchor_context_output.json")
    args = parser.parse_args()

    itinerary = _read_json_from_path_or_stdin(args.itinerary)
    pending = _read_json_from_path_or_stdin(args.pending_searches)
    result = resolve_anchor_context(
        itinerary,
        pending,
        destination_hint=args.destination_hint,
        start_anchor=args.start_anchor,
        end_anchor=args.end_anchor,
        cache_path=Path(args.cache_path),
    )

    if args.output_dir:
        # print(json.dumps(save_anchor_context_output(result, Path(args.output_dir)), ensure_ascii=False, indent=2))
        print(json.dumps(
            save_json_output(result, Path(args.output_dir), "03_anchor_context_output.json", "anchor_context"),
            ensure_ascii=False, 
            indent=2))
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
