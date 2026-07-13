import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
from dotenv import load_dotenv

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - keeps the search module usable without OpenAI installed.
    OpenAI = None


load_dotenv()

TEXT_SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"
DEFAULT_LANGUAGE_CODE = "zh-TW"
DEFAULT_REGION_CODE = "TW"
DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

BASE_FIELD_MASK = [
    "places.id",
    "places.name",
    "places.displayName",
    "places.formattedAddress",
    "places.location",
    "places.primaryType",
    "places.types",
    "places.googleMapsUri",
]

# These fields are useful for scoring, but Google bills them under higher Places SKUs.
QUALITY_FIELD_MASK = [
    "places.businessStatus",
    "places.currentOpeningHours",
    "places.regularOpeningHours",
    "places.rating",
    "places.userRatingCount",
    "places.websiteUri",
]


def _truthy_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


DEFAULT_LOCATION_BIAS_RADIUS_METERS = _float_env(
    "GOOGLE_PLACES_LOCATION_BIAS_RADIUS_METERS",
    1500.0,
)

COMMUTE_TERMS = {"預留通勤時間", "通勤時間", "交通時間"}
GENERIC_NEIGHBOR_HINTS = {
    "飯店",
    "酒店",
    "旅館",
    "住宿",
    "早餐",
    "午餐",
    "晚餐",
    "小吃",
    "美食",
    "返程",
    "準備返程",
}
LEADING_NEIGHBOR_HINT_PHRASES = [
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
    "入住酒店並享用",
    "入住酒店",
    "返回飯店並準備",
    "返回飯店",
    "返回酒店並準備",
    "返回酒店",
    "登上",
    "享用",
    "品嚐",
    "品嘗",
    "遊覽",
    "在",
]


def _compact_spaces(value: Any) -> str:
    return " ".join(str(value or "").split())


def _string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, list):
        items = value
    else:
        items = [value]

    cleaned: List[str] = []
    seen = set()
    for item in items:
        text = _compact_spaces(item)
        if not text or text in seen:
            continue
        seen.add(text)
        cleaned.append(text)
    return cleaned


def _merge_string_lists(*values: Any) -> List[str]:
    merged: List[str] = []
    seen = set()
    for value in values:
        for item in _string_list(value):
            if item in seen:
                continue
            seen.add(item)
            merged.append(item)
    return merged


def _normalize_text(value: Any) -> str:
    return _compact_spaces(value).replace("臺", "台").lower()


def _strip_parentheses(text: str) -> str:
    return re.sub(r"[（(].*?[）)]", "", text).strip()


def _strip_leading_neighbor_hint_phrases(text: str) -> str:
    current = text.strip()
    changed = True
    while changed:
        changed = False
        for phrase in LEADING_NEIGHBOR_HINT_PHRASES:
            if current.startswith(phrase) and len(current) > len(phrase):
                current = current[len(phrase) :].strip()
                changed = True
    return current


def _split_activity_components(text: str) -> List[str]:
    parts = re.split(r"和|與|及|、|/|／", text)
    cleaned = []
    for part in parts:
        item = _strip_leading_neighbor_hint_phrases(_strip_parentheses(part)).strip(" ，,。")
        if item:
            cleaned.append(item)
    return cleaned or [text]


def _looks_generic_neighbor_hint(text: Any) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return True
    if normalized in GENERIC_NEIGHBOR_HINTS:
        return True
    return any(term in normalized for term in ("預留通勤", "通勤時間"))


def _target_query_text(task: Dict[str, Any]) -> str:
    target_terms = _string_list(task.get("target_terms"))
    if target_terms:
        return " ".join(target_terms[:2])

    original = _compact_spaces(task.get("original_activity"))
    if original:
        return original

    return _compact_spaces(task.get("suggested_search_query"))


def _nearby_context_from_task(
    task: Dict[str, Any],
    nearby_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    context = dict(nearby_context) if isinstance(nearby_context, dict) else {}
    context.setdefault("previous_place_name", task.get("previous_place_hint"))
    context.setdefault("next_place_name", task.get("next_place_hint"))
    return context


def _normalize_region_code(value: Any) -> str:
    normalized = _compact_spaces(value).upper()
    return normalized if len(normalized) == 2 and normalized.isalpha() else ""


def _locale_from_trip_context(trip_context: Optional[Dict[str, Any]]) -> Tuple[str, str]:
    context = trip_context if isinstance(trip_context, dict) else {}
    language_code = (
        _compact_spaces(context.get("language_code"))
        or _compact_spaces(context.get("locale"))
        or os.getenv("GOOGLE_PLACES_LANGUAGE_CODE", DEFAULT_LANGUAGE_CODE)
    )
    region_code = (
        _normalize_region_code(context.get("region_code"))
        or _normalize_region_code(context.get("country_code"))
        or _normalize_region_code(context.get("destination_country_code"))
        or _normalize_region_code(os.getenv("GOOGLE_PLACES_REGION_CODE", DEFAULT_REGION_CODE))
    )
    return language_code, region_code


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
    raise TypeError("Expected a dict, list, JSON object string, or JSON array string.")


def _extract_pending_searches(raw: Any) -> List[Dict[str, Any]]:
    obj = _as_json_object(raw)
    pending = obj.get("pending_searches", obj)
    if not isinstance(pending, list):
        raise ValueError("Expected {'pending_searches': [...]} or a JSON array.")
    return [item for item in pending if isinstance(item, dict)]


def _itinerary_from_payload(
    payload: Dict[str, Any],
    explicit_itinerary: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    if isinstance(explicit_itinerary, dict):
        return explicit_itinerary

    for key in ("itinerary_json", "planner_result", "planner_output"):
        value = payload.get(key)
        if isinstance(value, dict):
            return value

    itinerary = payload.get("itinerary")
    if isinstance(itinerary, list):
        wrapped: Dict[str, Any] = {"itinerary": itinerary}
        trip_context = payload.get("trip_context")
        if isinstance(trip_context, dict):
            wrapped["trip_context"] = trip_context
        return wrapped

    return None


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


def _build_schedule_by_day(itinerary_json: Optional[Dict[str, Any]]) -> Dict[Any, List[Dict[str, Any]]]:
    if not isinstance(itinerary_json, dict):
        return {}
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


def _slot_key_for_item(item: Dict[str, Any], activity_key: str = "activity") -> str:
    return _slot_key(
        item.get("day"),
        item.get("time"),
        item.get(activity_key),
        slot_id=item.get("slot_id"),
    )


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


def _is_commute_item(item: Dict[str, Any]) -> bool:
    if item.get("is_commute") is True:
        return True
    if str(item.get("slot_type") or "").lower() == "transport":
        return True
    text = str(item.get("activity") or "")
    return any(term in text for term in COMMUTE_TERMS)


def _is_searchable_schedule_item(item: Dict[str, Any], searchable_keys: set) -> bool:
    return item.get("is_searchable") is True or _slot_key_for_item(item) in searchable_keys


def _nearest_neighbor_item(
    day_schedule: List[Dict[str, Any]],
    slot_index: int,
    direction: int,
    searchable_keys: set,
) -> Optional[Dict[str, Any]]:
    index = slot_index + direction
    while 0 <= index < len(day_schedule):
        item = day_schedule[index]
        if not _is_commute_item(item) and not _is_searchable_schedule_item(item, searchable_keys):
            return item
        index += direction
    return None


def _hint_from_schedule_item(item: Optional[Dict[str, Any]], role: str) -> str:
    if not isinstance(item, dict):
        return ""

    route_key = "route_exit_place" if role == "previous" else "route_entry_place"
    for key in (route_key, "place_query"):
        value = _compact_spaces(item.get(key))
        if value and not _looks_generic_neighbor_hint(value):
            return value

    activity = _strip_leading_neighbor_hint_phrases(
        _strip_parentheses(_compact_spaces(item.get("activity")))
    ).strip(" ，,。")
    components = _split_activity_components(activity)
    chosen = components[-1] if role == "previous" else components[0]
    chosen = _compact_spaces(chosen).strip(" ，,。")
    return "" if _looks_generic_neighbor_hint(chosen) else chosen


def _context_anchor_hint(context: Dict[str, Any], keys: Tuple[str, ...]) -> str:
    for key in keys:
        value = context.get(key)
        if isinstance(value, dict):
            for nested_key in ("place_query", "name", "address"):
                nested_value = _compact_spaces(value.get(nested_key))
                if nested_value:
                    return nested_value
        value_text = _compact_spaces(value)
        if value_text:
            return value_text
    return ""


def enrich_pending_searches_with_neighbor_hints(
    pending_searches_json: Any,
    itinerary_json: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Add lightweight previous/next place names for query expansion.

    This does not resolve anchors with Google Places and does not create coordinates.
    It only reads nearby schedule items from the planner itinerary.
    """
    pending_searches = _extract_pending_searches(pending_searches_json)
    schedule_by_day = _build_schedule_by_day(itinerary_json)
    if not schedule_by_day:
        return [dict(item) for item in pending_searches]

    searchable_keys = _pending_keys(pending_searches)
    context = itinerary_json.get("trip_context") if isinstance(itinerary_json, dict) else {}
    context = context if isinstance(context, dict) else {}
    start_hint = _context_anchor_hint(context, ("arrival_anchor", "start_anchor", "origin_anchor"))
    end_hint = _context_anchor_hint(context, ("return_anchor", "end_anchor", "departure_anchor"))
    day_order = list(schedule_by_day.keys())
    first_day = day_order[0] if day_order else None
    last_day = day_order[-1] if day_order else None

    enriched: List[Dict[str, Any]] = []
    for pending in pending_searches:
        item = dict(pending)
        day, slot_index = _find_slot_position(schedule_by_day, pending)
        day_schedule = _get_day_schedule(schedule_by_day, day) if day is not None else []
        neighbor_hints: Dict[str, Any] = {}

        if day is not None and slot_index is not None:
            previous_item = _nearest_neighbor_item(day_schedule, slot_index, -1, searchable_keys)
            next_item = _nearest_neighbor_item(day_schedule, slot_index, 1, searchable_keys)
            previous_hint = _hint_from_schedule_item(previous_item, "previous")
            next_hint = _hint_from_schedule_item(next_item, "next")

            if not previous_hint and str(day) == str(first_day):
                previous_hint = start_hint
            if not next_hint and str(day) == str(last_day):
                next_hint = end_hint

            if previous_hint and not item.get("previous_place_hint"):
                item["previous_place_hint"] = previous_hint
            if next_hint and not item.get("next_place_hint"):
                item["next_place_hint"] = next_hint

            neighbor_hints = {
                "source": "itinerary_neighbor_text",
                "previous": {
                    "slot_id": None if previous_item is None else previous_item.get("slot_id"),
                    "activity": None if previous_item is None else previous_item.get("activity"),
                    "place_hint": item.get("previous_place_hint"),
                },
                "next": {
                    "slot_id": None if next_item is None else next_item.get("slot_id"),
                    "activity": None if next_item is None else next_item.get("activity"),
                    "place_hint": item.get("next_place_hint"),
                },
            }

        if neighbor_hints:
            item["neighbor_hints"] = neighbor_hints
        enriched.append(item)

    return enriched


def _fallback_queries(
    task: Dict[str, Any],
    destination_hint: str = "",
    nearby_context: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    original = str(task.get("original_activity") or "").strip()
    suggested = str(task.get("suggested_search_query") or "").strip()
    reason = str(task.get("reason_for_search") or "").strip()
    target_text = _target_query_text(task)
    context = _nearby_context_from_task(task, nearby_context)
    base_items = []
    nearby_items = []
    intent_type = _compact_spaces(task.get("intent_type")) or reason or "places_search"
    task_target_terms = _string_list(task.get("target_terms")) or ([original] if original else [])
    task_location_terms = _string_list(task.get("location_terms"))
    if destination_hint and destination_hint not in task_location_terms:
        task_location_terms = [destination_hint] + task_location_terms

    if destination_hint and target_text:
        base_items.append(f"{destination_hint} {target_text}")

    for hint_key in ("previous_place_name", "next_place_name"):
        hint = _compact_spaces(context.get(hint_key))
        if not hint or not target_text:
            continue
        nearby_term = hint if hint.endswith("附近") else f"{hint}附近"
        nearby_items.append(f"{nearby_term} {target_text}")
        if nearby_term not in task_location_terms:
            task_location_terms.append(nearby_term)

    base_items.extend(nearby_items)
    base_items.extend([suggested, original])

    if destination_hint and original and destination_hint not in original:
        base_items.append(f"{destination_hint} {original}")
    if destination_hint and suggested and destination_hint not in suggested:
        base_items.append(f"{destination_hint} {suggested}")

    queries: List[Dict[str, str]] = []
    seen = set()
    for item in base_items:
        normalized = " ".join(item.split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        queries.append(
            {
                "query": normalized,
                "intent": intent_type,
                "intent_type": intent_type,
                "target_terms": task_target_terms,
                "location_terms": task_location_terms,
                "must_have": _string_list(task.get("must_have")),
            }
        )
    return queries[:4]


def generate_places_queries(
    task: Dict[str, Any],
    destination_hint: str = "",
    max_queries: int = 4,
    model: str = DEFAULT_OPENAI_MODEL,
    trip_context: Optional[Dict[str, Any]] = None,
    nearby_context: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Use an LLM to turn one fuzzy itinerary slot into several short Google Places queries.
    Falls back to the extractor's suggested_search_query if OpenAI is unavailable.
    """
    resolved_nearby_context = _nearby_context_from_task(task, nearby_context)
    if OpenAI is None or not os.getenv("OPENAI_API_KEY"):
        return _fallback_queries(
            task,
            destination_hint=destination_hint,
            nearby_context=resolved_nearby_context,
        )

    system_prompt = """
你是 Google Places Text Search 查詢產生器。
你的任務是把旅遊行程中的模糊活動，轉成 3 到 4 組適合 Google Places API 搜尋的短查詢。

核心策略：
1. 第一類 query：用目的地 + 目標類型做廣泛搜尋，例如「台南 牛肉湯」。
2. 第二類 query：如果 nearby_context 有 previous_place_name 或 next_place_name，產生附近搜尋，例如「赤崁樓附近 牛肉湯」。
3. 第三類 query：如果活動有明確時段或限制，可加入必要語意，例如「台南 午餐 牛肉湯」。
4. 前後地點只能當作 location hint，不要把前後地點當成要搜尋的主體。
5. 查詢要短，不要產生文章式搜尋，例如「推薦文章」、「PTT」、「排行」、「攻略」。
6. 不要輸出已知景點本身，除非 ambiguous_slot 本來就是要找該景點。
7. 只能輸出合法 JSON，不要 Markdown。

每一筆 query 都要附上結構化搜尋線索：
- target_terms：真正要找的主體，例如 ["牛肉湯"]、["老屋咖啡"]、["夜市"]。
- location_terms：城市、行政區、街區或附近 anchor，例如 ["台南"]、["赤崁樓附近"]。
- intent_type：restaurant / cafe / museum / attraction / local_culture / shopping / night_market / nature / places_search 等。
- must_have：該候選必須滿足的限制，例如「適合午餐」「可停留兩小時」。沒有就空陣列。

輸出格式：
{
  "queries": [
    {
      "query": "台南 牛肉湯",
      "intent": "restaurant",
      "intent_type": "restaurant",
      "target_terms": ["牛肉湯"],
      "location_terms": ["台南"],
      "must_have": []
    },
    {
      "query": "赤崁樓附近 牛肉湯",
      "intent": "restaurant",
      "intent_type": "restaurant",
      "target_terms": ["牛肉湯"],
      "location_terms": ["赤崁樓附近"],
      "must_have": []
    }
  ]
}
"""

    user_payload = {
        "destination_hint": destination_hint,
        "trip_context": trip_context or {},
        "nearby_context": resolved_nearby_context,
        "max_queries": max_queries,
        "ambiguous_slot": {
            "slot_id": task.get("slot_id"),
            "day": task.get("day"),
            "start_time": task.get("start_time"),
            "end_time": task.get("end_time"),
            "time": task.get("time"),
            "slot_type": task.get("slot_type"),
            "original_activity": task.get("original_activity"),
            "reason_for_search": task.get("reason_for_search"),
            "suggested_search_query": task.get("suggested_search_query"),
            "target_terms": task.get("target_terms"),
            "location_terms": task.get("location_terms"),
            "intent_type": task.get("intent_type"),
            "must_have": task.get("must_have"),
        },
    }

    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        parsed = json.loads(content)
        queries = parsed.get("queries", [])
        cleaned: List[Dict[str, str]] = []
        seen = set()
        for item in queries:
            if not isinstance(item, dict):
                continue
            query = " ".join(str(item.get("query") or "").split())
            if not query or query in seen:
                continue
            seen.add(query)
            intent_type = str(item.get("intent_type") or item.get("intent") or "places_search")
            cleaned.append(
                {
                    "query": query,
                    "intent": str(item.get("intent") or intent_type),
                    "intent_type": intent_type,
                    "target_terms": _string_list(item.get("target_terms")),
                    "location_terms": _string_list(item.get("location_terms")),
                    "must_have": _string_list(item.get("must_have")),
                }
            )
        if cleaned:
            return cleaned[:max_queries]
    except Exception:
        pass

    return _fallback_queries(
        task,
        destination_hint=destination_hint,
        nearby_context=resolved_nearby_context,
    )


class GooglePlacesTextSearch:
    def __init__(
        self,
        api_key: Optional[str] = None,
        include_quality_fields: Optional[bool] = None,
        timeout_seconds: int = 15,
    ) -> None:
        self.api_key = (
            api_key
            or os.getenv("GOOGLE_PLACES_API_KEY")
            or os.getenv("GOOGLE_MAPS_API_KEY")
        )
        self.include_quality_fields = (
            _truthy_env("GOOGLE_PLACES_INCLUDE_QUALITY_FIELDS")
            if include_quality_fields is None
            else include_quality_fields
        )
        self.timeout_seconds = timeout_seconds

    def _field_mask(self) -> str:
        fields = list(BASE_FIELD_MASK)
        if self.include_quality_fields:
            fields.extend(QUALITY_FIELD_MASK)
        return ",".join(fields)

    def search_text(
        self,
        query: str,
        page_size: int = 6,
        language_code: str = DEFAULT_LANGUAGE_CODE,
        region_code: str = DEFAULT_REGION_CODE,
        location_bias: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if not self.api_key:
            raise RuntimeError(
                "Missing Google Places API key. Set GOOGLE_PLACES_API_KEY "
                "or GOOGLE_MAPS_API_KEY in .env."
            )

        safe_page_size = max(1, min(int(page_size), 20))
        body: Dict[str, Any] = {
            "textQuery": query,
            "pageSize": safe_page_size,
            "languageCode": language_code,
            "regionCode": region_code,
        }
        if location_bias:
            body["locationBias"] = location_bias

        response = requests.post(
            TEXT_SEARCH_URL,
            headers={
                "Content-Type": "application/json",
                "X-Goog-Api-Key": self.api_key,
                "X-Goog-FieldMask": self._field_mask(),
            },
            json=body,
            timeout=self.timeout_seconds,
        )

        if response.status_code >= 400:
            raise RuntimeError(
                f"Google Places Text Search failed ({response.status_code}): "
                f"{response.text[:800]}"
            )

        data = response.json()
        return data.get("places", [])


def _display_name(place: Dict[str, Any]) -> str:
    display = place.get("displayName")
    if isinstance(display, dict):
        return str(display.get("text") or "")
    return ""


def _place_id(place: Dict[str, Any]) -> str:
    if place.get("id"):
        return str(place["id"])
    resource_name = str(place.get("name") or "")
    if resource_name.startswith("places/"):
        return resource_name.split("/", 1)[1]
    return ""


def _normalize_place(
    place: Dict[str, Any],
    query: str,
    query_intent: str,
    rank_index: int,
    query_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    location = place.get("location") or {}
    rating = place.get("rating")
    user_rating_count = place.get("userRatingCount")
    google_rank_score = round(1.0 / (rank_index + 1), 4)
    metadata = query_metadata if isinstance(query_metadata, dict) else {}
    matched_query = {
        "query": query,
        "intent": query_intent,
        "intent_type": metadata.get("intent_type") or query_intent,
        "rank": rank_index + 1,
        "target_terms": _string_list(metadata.get("target_terms")),
        "location_terms": _string_list(metadata.get("location_terms")),
        "must_have": _string_list(metadata.get("must_have")),
    }

    normalized: Dict[str, Any] = {
        "place_id": _place_id(place),
        "resource_name": place.get("name"),
        "name": _display_name(place),
        "address": place.get("formattedAddress"),
        "latitude": location.get("latitude"),
        "longitude": location.get("longitude"),
        "primary_type": place.get("primaryType"),
        "types": place.get("types", []),
        "google_maps_uri": place.get("googleMapsUri"),
        "matched_queries": [matched_query],
        "ranking_signals": {
            "google_rank_score": google_rank_score,
        },
    }

    if rating is not None:
        normalized["rating"] = rating
        normalized["ranking_signals"]["rating_score"] = round(float(rating) / 5.0, 4)
    if user_rating_count is not None:
        normalized["user_rating_count"] = user_rating_count
    if place.get("businessStatus"):
        normalized["business_status"] = place.get("businessStatus")
    if place.get("currentOpeningHours"):
        normalized["current_opening_hours"] = place.get("currentOpeningHours")
    if place.get("regularOpeningHours"):
        normalized["regular_opening_hours"] = place.get("regularOpeningHours")
    if place.get("websiteUri"):
        normalized["website_uri"] = place.get("websiteUri")

    return normalized


def _slot_key(day: Any, time: Any, activity: Any, slot_id: Any = None) -> str:
    if slot_id:
        return f"slot_id:{slot_id}"
    return "|".join(
        [
            str(day or ""),
            str(time or ""),
            _compact_spaces(activity).replace("臺", "台").lower(),
        ]
    )


def _coordinates(place: Optional[Dict[str, Any]]) -> Optional[Tuple[float, float]]:
    if not isinstance(place, dict):
        return None
    latitude = place.get("latitude")
    longitude = place.get("longitude")
    if latitude is None or longitude is None:
        location = place.get("location") if isinstance(place.get("location"), dict) else {}
        latitude = location.get("latitude")
        longitude = location.get("longitude")
    if latitude is None or longitude is None:
        return None
    try:
        return float(latitude), float(longitude)
    except (TypeError, ValueError):
        return None


def _anchor_place(anchor: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(anchor, dict):
        return None
    place = anchor.get("place")
    return place if isinstance(place, dict) else None


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


def _location_bias_from_anchor_context(
    slot: Dict[str, Any],
    anchors_by_slot: Dict[str, Dict[str, Any]],
    radius_meters: float = DEFAULT_LOCATION_BIAS_RADIUS_METERS,
) -> Optional[Dict[str, Any]]:
    slot_anchor = anchors_by_slot.get(
        _slot_key(
            slot.get("day"),
            slot.get("time"),
            slot.get("original_activity"),
            slot_id=slot.get("slot_id"),
        )
    )
    if not slot_anchor:
        return None

    coordinates = [
        point
        for point in [
            _coordinates(_anchor_place(slot_anchor.get("previous_anchor"))),
            _coordinates(_anchor_place(slot_anchor.get("next_anchor"))),
        ]
        if point is not None
    ]
    if not coordinates:
        return None

    center_latitude = sum(point[0] for point in coordinates) / len(coordinates)
    center_longitude = sum(point[1] for point in coordinates) / len(coordinates)
    return {
        "circle": {
            "center": {
                "latitude": round(center_latitude, 7),
                "longitude": round(center_longitude, 7),
            },
            "radius": float(radius_meters),
        }
    }


def _merge_candidates(candidates: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    ordered_keys: List[str] = []

    for candidate in candidates:
        key = candidate.get("place_id") or f"{candidate.get('name')}|{candidate.get('address')}"
        if key not in merged:
            merged[key] = candidate
            ordered_keys.append(key)
            continue

        existing = merged[key]
        existing["matched_queries"].extend(candidate.get("matched_queries", []))
        old_score = existing.get("ranking_signals", {}).get("google_rank_score", 0)
        new_score = candidate.get("ranking_signals", {}).get("google_rank_score", 0)
        existing["ranking_signals"]["google_rank_score"] = max(old_score, new_score)

    return [merged[key] for key in ordered_keys]


def _collect_query_metadata(queries: Iterable[Dict[str, Any]], key: str) -> List[str]:
    collected: List[str] = []
    seen = set()
    for query in queries:
        if not isinstance(query, dict):
            continue
        for item in _string_list(query.get(key)):
            if item in seen:
                continue
            seen.add(item)
            collected.append(item)
    return collected


def search_places_for_pending_searches(
    pending_searches_json: Any,
    destination_hint: str = "",
    max_queries_per_slot: int = 4,
    max_results_per_query: int = 6,
    include_quality_fields: Optional[bool] = None,
    expand_queries_with_llm: bool = True,
    trip_context: Optional[Dict[str, Any]] = None,
    anchor_context: Optional[Dict[str, Any]] = None,
    itinerary_json: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = _as_json_object(pending_searches_json)
    pending_searches = _extract_pending_searches(payload)
    resolved_itinerary_json = _itinerary_from_payload(payload, itinerary_json)
    pending_searches = enrich_pending_searches_with_neighbor_hints(
        pending_searches,
        itinerary_json=resolved_itinerary_json,
    )
    resolved_trip_context = (
        trip_context
        if isinstance(trip_context, dict)
        else payload.get("trip_context")
        if isinstance(payload.get("trip_context"), dict)
        else {}
    )
    if not resolved_trip_context and isinstance(resolved_itinerary_json, dict):
        resolved_trip_context = resolved_itinerary_json.get("trip_context") or {}
    if not resolved_trip_context and isinstance(anchor_context, dict):
        resolved_trip_context = anchor_context.get("trip_context") or {}
    if not isinstance(resolved_trip_context, dict):
        resolved_trip_context = {}
    language_code, region_code = _locale_from_trip_context(resolved_trip_context)
    resolved_destination_hint = (
        destination_hint
        or _compact_spaces(resolved_trip_context.get("destination"))
        or _compact_spaces(resolved_trip_context.get("city"))
    )
    resolved_anchor_context = (
        anchor_context
        if isinstance(anchor_context, dict)
        else payload.get("anchor_context")
        if isinstance(payload.get("anchor_context"), dict)
        else None
    )
    anchors_by_slot = _anchor_context_by_slot(resolved_anchor_context)
    searcher = GooglePlacesTextSearch(include_quality_fields=include_quality_fields)
    slots: List[Dict[str, Any]] = []

    for task in pending_searches:
        if expand_queries_with_llm:
            queries = generate_places_queries(
                task,
                destination_hint=resolved_destination_hint,
                max_queries=max_queries_per_slot,
                trip_context=resolved_trip_context,
                nearby_context={
                    "previous_place_name": task.get("previous_place_hint"),
                    "next_place_name": task.get("next_place_hint"),
                },
            )
        else:
            queries = _fallback_queries(
                task,
                destination_hint=resolved_destination_hint,
                nearby_context={
                    "previous_place_name": task.get("previous_place_hint"),
                    "next_place_name": task.get("next_place_hint"),
                },
            )

        raw_candidates: List[Dict[str, Any]] = []
        errors: List[Dict[str, str]] = []
        location_bias = _location_bias_from_anchor_context(task, anchors_by_slot)
        for query_item in queries:
            query = query_item["query"]
            intent = query_item.get("intent", "places_search")
            try:
                places = searcher.search_text(
                    query,
                    page_size=max_results_per_query,
                    language_code=language_code,
                    region_code=region_code,
                    location_bias=location_bias,
                )
                for index, place in enumerate(places):
                    raw_candidates.append(_normalize_place(place, query, intent, index, query_item))
            except Exception as exc:
                errors.append({"query": query, "error": str(exc)})
        slot_target_terms = _merge_string_lists(
            task.get("target_terms"),
            _collect_query_metadata(queries, "target_terms"),
        )
        slot_location_terms = _merge_string_lists(
            task.get("location_terms"),
            _collect_query_metadata(queries, "location_terms"),
        )
        slot_must_have = _merge_string_lists(
            task.get("must_have"),
            _collect_query_metadata(queries, "must_have"),
        )
        slot_intent_type = _compact_spaces(task.get("intent_type"))
        if not slot_intent_type:
            for query_item in queries:
                slot_intent_type = _compact_spaces(query_item.get("intent_type") or query_item.get("intent"))
                if slot_intent_type:
                    break

        slots.append(
            {
                "slot_id": task.get("slot_id"),
                "day": task.get("day"),
                "day_of_week": task.get("day_of_week"),
                "start_time": task.get("start_time"),
                "end_time": task.get("end_time"),
                "time": task.get("time"),
                "slot_type": task.get("slot_type"),
                "original_activity": task.get("original_activity"),
                "reason_for_search": task.get("reason_for_search"),
                "previous_place_hint": task.get("previous_place_hint"),
                "next_place_hint": task.get("next_place_hint"),
                "neighbor_hints": task.get("neighbor_hints"),
                "target_terms": slot_target_terms,
                "location_terms": slot_location_terms,
                "intent_type": slot_intent_type or None,
                "must_have": slot_must_have,
                "queries": queries,
                "search_locale": {
                    "language_code": language_code,
                    "region_code": region_code,
                },
                "location_bias": location_bias,
                "candidates": _merge_candidates(raw_candidates),
                "errors": errors,
            }
        )

    return {
        "places_search_version": "places_search_agent_v0.3",
        "trip_context": resolved_trip_context,
        "destination_hint": resolved_destination_hint,
        "search_locale": {
            "language_code": language_code,
            "region_code": region_code,
        },
        "candidate_slots": slots,
    }


def build_places_query_preview(
    pending_searches_json: Any,
    itinerary_json: Dict[str, Any],
    destination_hint: str = "",
    max_queries_per_slot: int = 4,
) -> Dict[str, Any]:
    pending_searches = enrich_pending_searches_with_neighbor_hints(
        pending_searches_json,
        itinerary_json=itinerary_json,
    )
    trip_context = itinerary_json.get("trip_context")
    trip_context = trip_context if isinstance(trip_context, dict) else {}
    resolved_destination_hint = (
        destination_hint
        or _compact_spaces(trip_context.get("destination"))
        or _compact_spaces(trip_context.get("city"))
    )
    language_code, region_code = _locale_from_trip_context(trip_context)

    slots = []
    for task in pending_searches:
        queries = _fallback_queries(
            task,
            destination_hint=resolved_destination_hint,
            nearby_context={
                "previous_place_name": task.get("previous_place_hint"),
                "next_place_name": task.get("next_place_hint"),
            },
        )[:max_queries_per_slot]
        slots.append(
            {
                "slot_id": task.get("slot_id"),
                "day": task.get("day"),
                "time": task.get("time"),
                "slot_type": task.get("slot_type"),
                "original_activity": task.get("original_activity"),
                "previous_place_hint": task.get("previous_place_hint"),
                "next_place_hint": task.get("next_place_hint"),
                "neighbor_hints": task.get("neighbor_hints"),
                "queries": queries,
                "candidates": [],
            }
        )

    return {
        "places_search_version": "places_query_preview_v0.1",
        "dry_run_queries": True,
        "trip_context": trip_context,
        "destination_hint": resolved_destination_hint,
        "search_locale": {
            "language_code": language_code,
            "region_code": region_code,
        },
        "candidate_slots": slots,
    }


def save_json_output(data: Dict[str, Any], output_dir: Path, filename: str, result_key: str) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return {result_key: str(path)}


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent

    input_path = (
        Path(sys.argv[1]).resolve()
        if len(sys.argv) > 1
        else base_dir / "traced_outputs" / "北京上海測資_result.json"
    )
    stage1_path = (
        Path(sys.argv[2]).resolve()
        if len(sys.argv) > 2
        else input_path.parent / f"{input_path.stem}_stage1.json"
    )

    with open(input_path, "r", encoding="utf-8") as f:
        itinerary_context = json.load(f)
    with open(stage1_path, "r", encoding="utf-8") as f:
        pending_context = json.load(f)

    RUN_GOOGLE_PLACES_SEARCH = True

    # print("=" * 20 + "看一下輸入檔" + "=" * 20)
    # print("planner:", input_path)
    # print("stage1:", stage1_path)
    # print("=" * 20 + "結束" + "=" * 20)
    # print("\n")

    if RUN_GOOGLE_PLACES_SEARCH:
        places_result = search_places_for_pending_searches(
            pending_context,
            destination_hint=itinerary_context.get("trip_context", {}).get("destination", ""),
            max_queries_per_slot=4,
            max_results_per_query=6,
            expand_queries_with_llm=True,
            trip_context=itinerary_context.get("trip_context"),
            itinerary_json=itinerary_context,
        )
        append_text = "places_search"
    else:
        places_result = build_places_query_preview(
            pending_context,
            itinerary_context,
            max_queries_per_slot=4,
        )
        append_text = "places_query_preview"

    print("=" * 20 + "看一下搜尋詞" + "=" * 20)
    for slot in places_result["candidate_slots"]:
        print(slot.get("slot_id"), slot.get("original_activity"))
        print("previous:", slot.get("previous_place_hint"))
        print("next:", slot.get("next_place_hint"))
        for query in slot.get("queries", []):
            print("-", query.get("query"))
        print()
    print("=" * 20 + "結束" + "=" * 20)
    print("\n")

    filename = f"{input_path.stem}_{append_text}.json"
    output_path = base_dir / "traced_outputs"
    saved_path = save_json_output(places_result, output_path, filename, "places_search")

    # print("=" * 20 + "看一下保存位置" + "=" * 20)
    # print(saved_path)
    # print("=" * 20 + "結束" + "=" * 20)
