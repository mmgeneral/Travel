# corporate_with_gemini/LodgingSearchAgent.py 參考版
from __future__ import annotations

import os
import re
import requests
import sys
import json
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional
from AnchorResolver import save_json_output
from dotenv import load_dotenv

"""
這隻程式主要步驟就下面兩步
ctx = build_lodging_search_context(planner_result) ctx = 組出適合的搜尋字詞
lodging_result = search_lodging_serpapi(ctx, max_results=20) 把ctx送去SerpApi取得飯店資料
"""

load_dotenv(override=True)
api_key = os.getenv("SERPAPI_API_KEY")

SERPAPI_URL = "https://serpapi.com/search.json"
DEFAULT_BUDGET_CURRENCY = "CNY"
SEARCH_PRICE_CURRENCY = "TWD"
TWD_PER_CNY = 5.0
LODGING_BUDGET_SHARE = 0.40

# True: 把 SerpApi 回來的飯店價格轉成預算幣別一起保存，預設方便用人民幣預算做比較。
CONVERT_LODGING_PRICE_TO_BUDGET_CURRENCY = True
CONVERT_BUDGET_CAP_TO_SEARCH_CURRENCY = True


def _default_checkin_date(days_after_today: int = 3) -> str:
    return (date.today() + timedelta(days=days_after_today)).isoformat()

def _anchor_query(anchor: Any) -> str:
    if isinstance(anchor, dict):
        return _compact(anchor.get("place_query") or anchor.get("name") or "")
    return _compact(anchor)


def _extract_lodging_area(trip: Dict[str, Any], where: Dict[str, Any], what: Dict[str, Any]) -> str:
    lodging_anchor = _anchor_query(trip.get("lodging_anchor"))
    if lodging_anchor:
        return lodging_anchor

    lodging_preferences = what.get("lodging_preferences") or []
    for pref in lodging_preferences:
        text = _compact(pref)
        if any(marker in text for marker in ["住", "住宿", "附近", "車站", "站", "區", "商圈"]):
            cleaned = (
                text.replace("希望住", "")
                .replace("想住", "")
                .replace("住宿", "")
                .replace("附近", "")
                .strip()
            )
            if cleaned:
                return cleaned

    # 「交通方便」不一定是地名，這時用抵達 anchor 當住宿搜尋中心。
    joined_prefs = " ".join(_compact(item) for item in lodging_preferences)
    if any(term in joined_prefs for term in ["交通方便", "近車站", "車站附近", "方便移動"]):
        arrival_anchor = _anchor_query(trip.get("arrival_anchor"))
        if arrival_anchor:
            return arrival_anchor

    return ""

def _infer_lodging_type(what: Dict[str, Any]) -> str:
    prefs = " ".join(_compact(item).lower() for item in (what.get("lodging_preferences") or []))

    if any(term in prefs for term in ["hostel", "青年旅館", "背包", "膠囊"]):
        return "青年旅館"
    if any(term in prefs for term in ["商務", "business"]):
        return "商務飯店"
    if any(term in prefs for term in ["溫泉", "ryokan", "旅館"]):
        return "旅館"
    if any(term in prefs for term in ["resort", "度假"]):
        return "度假飯店"

    return "飯店"

def _build_lodging_query(destination: str, lodging_area: str, lodging_type: str) -> str:
    parts = [destination]

    if lodging_area and destination not in lodging_area:
        parts.append(lodging_area)
    elif lodging_area:
        parts.append(lodging_area)

    parts.append(lodging_type)

    return _compact(" ".join(parts))

def _compact(value: Any) -> str:
    return " ".join(str(value or "").split())


def _parse_budget(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    text = _compact(value).replace(",", "")
    match = re.search(r"\d+(?:\.\d+)?", text)
    return float(match.group()) if match else None


def _normalize_currency(value: Any, default: str = DEFAULT_BUDGET_CURRENCY) -> str:
    text = _compact(value).upper()
    if text in {"RMB", "CNY", "人民幣", "人民币", "元人民幣", "元人民币"}:
        return "CNY"
    if text in {"TWD", "NTD", "NT$", "新台幣", "新臺幣", "台幣", "臺幣"}:
        return "TWD"
    return text or default


def _infer_currency_from_text(value: Any, default: str = DEFAULT_BUDGET_CURRENCY) -> str:
    text = _compact(value)
    if any(term in text for term in ["人民幣", "人民币", "RMB", "CNY"]):
        return "CNY"
    if any(term in text for term in ["新台幣", "新臺幣", "台幣", "臺幣", "TWD", "NTD", "NT$"]):
        return "TWD"
    return default


def _parse_budget_amount_and_currency(how: Dict[str, Any]) -> tuple[Optional[float], str]:
    raw_budget = how.get("budget")
    explicit_currency = how.get("budget_currency")

    if isinstance(raw_budget, dict):
        amount = _parse_budget(
            raw_budget.get("amount")
            or raw_budget.get("total")
            or raw_budget.get("value")
        )
        currency = _normalize_currency(raw_budget.get("currency") or explicit_currency)
        return amount, currency

    amount = _parse_budget(raw_budget)
    currency = _normalize_currency(explicit_currency or _infer_currency_from_text(raw_budget))
    return amount, currency


def _convert_money(amount: Optional[float], from_currency: str, to_currency: str) -> Optional[float]:
    if amount is None:
        return None

    source = _normalize_currency(from_currency)
    target = _normalize_currency(to_currency)
    if source == target:
        return round(float(amount), 2)
    if source == "CNY" and target == "TWD":
        return round(float(amount) * TWD_PER_CNY, 2)
    if source == "TWD" and target == "CNY":
        return round(float(amount) / TWD_PER_CNY, 2)
    return round(float(amount), 2)


def _date_add_days(date_text: str, days: int) -> str:
    start = datetime.strptime(date_text, "%Y-%m-%d").date()
    return (start + timedelta(days=days)).isoformat()


def build_lodging_search_context(planner_result: Dict[str, Any]) -> Dict[str, Any]:
    trip = planner_result.get("trip_context") or {}
    brief = planner_result.get("planning_brief_5w1h") or {}

    who = brief.get("who") or {}
    where = brief.get("where") or {}
    when = brief.get("when") or {}
    what = brief.get("what") or {}
    why = brief.get("why") or {}
    how = brief.get("how") or {}

    destination = _compact(where.get("destination") or trip.get("destination") or trip.get("city"))
    lodging_area = _extract_lodging_area(trip, where, what)
    lodging_type = _infer_lodging_type(what)
    query = _build_lodging_query(destination, lodging_area, lodging_type)


    # preferred_areas = (
    #     where.get("must_include_areas")
    #     or where.get("key_areas")
    #     or []
    # )
    # avoid_areas = where.get("avoid_areas") or []

    checkin_date = (
        when.get("travel_start_date")
        or trip.get("travel_start_date")
    )
    if not checkin_date:
        checkin_date = _default_checkin_date(3)
        date_source = "default_today_plus_3"

    nights = when.get("duration_nights")
    try:
        nights = int(nights) if nights is not None else None
    except (TypeError, ValueError):
        nights = None

    checkout_date = when.get("travel_end_date")
    if checkin_date and not checkout_date and nights:
        checkout_date = _date_add_days(checkin_date, nights)

    traveler_count = who.get("traveler_count")
    try:
        adults = max(1, int(traveler_count or 2))
    except (TypeError, ValueError):
        adults = 2

    budget_total, budget_currency = _parse_budget_amount_and_currency(how)
    budget_for_lodging_total = round(budget_total * LODGING_BUDGET_SHARE, 2) if budget_total else None
    budget_per_night_cap = (
        round(budget_for_lodging_total / nights, 2)
        if budget_for_lodging_total and nights
        else None
    )
    search_currency = SEARCH_PRICE_CURRENCY
    budget_per_night_cap_search_currency = (
        _convert_money(budget_per_night_cap, budget_currency, search_currency)
        if CONVERT_BUDGET_CAP_TO_SEARCH_CURRENCY
        else budget_per_night_cap
    )

    origin_country = _compact(trip.get("origin_country_code") or "TW").lower()
    language_code = _compact(trip.get("language_code") or "zh-TW").lower()
    # lodging_preferences = what.get("lodging_preferences") or []
    # priority_order = why.get("priority_order") or []

    # primary_area = _compact(preferred_areas[0]) if preferred_areas else ""
    # query_parts = [destination, primary_area, "飯店"]
    # if lodging_preferences:
    #     query_parts.append(_compact(lodging_preferences[0]))

    return {
        "destination": destination,
        "lodging_area": lodging_area,
        "lodging_type": lodging_type,
        "query": query,
        "checkin_date": checkin_date,
        "checkout_date": checkout_date,
        "nights": nights,
        "adults": adults,
        "currency": search_currency,
        "gl": origin_country,
        "hl": language_code,
        "budget_total": budget_total,
        "budget_currency": budget_currency,
        "lodging_budget_share": LODGING_BUDGET_SHARE,
        "budget_for_lodging_total": budget_for_lodging_total,
        "budget_per_night_cap": budget_per_night_cap,
        "budget_per_night_cap_search_currency": budget_per_night_cap_search_currency,
        "max_price": budget_per_night_cap_search_currency,
        "convert_lodging_price_to_budget_currency": CONVERT_LODGING_PRICE_TO_BUDGET_CURRENCY,
        "lodging_preferences": what.get("lodging_preferences") or [],
        "avoid_areas": where.get("avoid_areas") or [],
        "priority_order": why.get("priority_order") or [],
        "can_search_live_price": bool(checkin_date and checkout_date),
    }


def _price_value(rate: Dict[str, Any]) -> Optional[float]:
    for key in ("extracted_lowest", "extracted_before_taxes_fees"):
        value = rate.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def normalize_serpapi_hotel(prop: Dict[str, Any], rank: int, ctx: Dict[str, Any]) -> Dict[str, Any]:
    gps = prop.get("gps_coordinates") or {}
    rate = prop.get("rate_per_night") or {}
    total = prop.get("total_rate") or {}

    per_night = _price_value(rate)
    total_price = _price_value(total)
    nights = max(1, int(ctx.get("nights") or 1))

    if per_night is None and total_price is not None:
        per_night = round(total_price / nights, 2)
    if total_price is None and per_night is not None:
        total_price = round(per_night * nights, 2)

    budget_currency = ctx.get("budget_currency", DEFAULT_BUDGET_CURRENCY)
    price_currency = ctx.get("currency", SEARCH_PRICE_CURRENCY)
    per_night_budget_currency = None
    total_budget_currency = None
    if ctx.get("convert_lodging_price_to_budget_currency", CONVERT_LODGING_PRICE_TO_BUDGET_CURRENCY):
        per_night_budget_currency = _convert_money(per_night, price_currency, budget_currency)
        total_budget_currency = _convert_money(total_price, price_currency, budget_currency)

    return {
        "candidate_domain": "lodging",
        "provider": "serpapi_google_hotels",
        "place_id": prop.get("property_token") or prop.get("name"),
        "property_token": prop.get("property_token"),
        "name": prop.get("name"),
        "address": prop.get("address"),
        "latitude": gps.get("latitude"),
        "longitude": gps.get("longitude"),
        "rating": prop.get("overall_rating"),
        "user_rating_count": prop.get("reviews"),
        "hotel_class": prop.get("hotel_class"),
        "amenities": prop.get("amenities", []),
        "thumbnail": prop.get("thumbnail"),
        "price": {
            "currency": price_currency,
            "per_night": per_night,
            "total": total_price,
            "nights": nights,
            "budget_currency": budget_currency,
            "per_night_budget_currency": per_night_budget_currency,
            "total_budget_currency": total_budget_currency,
            "budget_per_night_cap": ctx.get("budget_per_night_cap"),
            "budget_for_lodging_total": ctx.get("budget_for_lodging_total"),
            "source": "serpapi_google_hotels",
            "availability_confirmed": per_night is not None or total_price is not None,
            "raw_rate_per_night": rate,
            "raw_total_rate": total,
        },
        "matched_queries": [{
            "query": ctx["query"],
            "intent_type": "lodging",
            "rank": rank,
            "target_terms": ["lodging"],
            "location_terms": [ctx["destination"]],
            "must_have": ctx.get("lodging_preferences", []),
        }],
        "ranking_signals": {"google_rank_score": round(1 / rank, 4)},
        "raw_candidate": prop,
    }


def _candidate_within_price_cap(candidate: Dict[str, Any], ctx: Dict[str, Any]) -> bool:
    cap = ctx.get("budget_per_night_cap")
    total_cap = ctx.get("budget_for_lodging_total")
    if not cap and not total_cap:
        return True

    price = candidate.get("price") or {}
    per_night = price.get("per_night_budget_currency")
    total = price.get("total_budget_currency")
    nights = max(1, int(price.get("nights") or ctx.get("nights") or 1))

    if per_night is None and total is not None:
        per_night = round(float(total) / nights, 2)
    if per_night is None:
        return False
    if cap and float(per_night) > float(cap):
        return False
    if total_cap and total is not None and float(total) > float(total_cap):
        return False
    return True


def search_lodging_serpapi(ctx: Dict[str, Any], max_results: int = 20) -> Dict[str, Any]:
    if not ctx.get("can_search_live_price"):
        return {
            "candidate_domain": "lodging",
            "status": "date_required_for_live_quote",
            "message": "缺少 checkin_date/checkout_date，無法查即時飯店價格。",
            "search_context": ctx,
            "candidates": [],
        }

    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing SERPAPI_API_KEY")

    params = {
        "engine": "google_hotels",
        "q": ctx["query"],
        "check_in_date": ctx["checkin_date"],
        "check_out_date": ctx["checkout_date"],
        "adults": ctx.get("adults", 2),
        "currency": ctx.get("currency", "TWD"),
        "gl": ctx.get("gl", "tw"),
        "hl": ctx.get("hl", "zh-tw"),
        "sort_by": 3,
        "api_key": api_key,
    }
    if ctx.get("max_price"):
        params["max_price"] = int(float(ctx["max_price"]))

    response = requests.get(SERPAPI_URL, params=params, timeout=30)
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        try:
            detail = response.json()
        except ValueError:
            detail = response.text[:500]
        raise RuntimeError(f"SerpApi hotel search failed: status={response.status_code}, detail={detail}") from exc
    data = response.json()
    if data.get("error"):
        raise RuntimeError(data["error"])

    properties = data.get("properties") or []
    candidates = [
        normalize_serpapi_hotel(prop, rank=i + 1, ctx=ctx)
        for i, prop in enumerate(properties[:max_results])
    ]
    filtered_candidates = [
        candidate
        for candidate in candidates
        if _candidate_within_price_cap(candidate, ctx)
    ]

    return {
        "lodging_search_version": "serpapi_google_hotels_v0.1",
        "candidate_domain": "lodging",
        "search_context": ctx,
        "candidate_count_before_price_filter": len(candidates),
        "candidate_count": len(filtered_candidates),
        "price_filter": {
            "enabled": bool(ctx.get("budget_per_night_cap")),
            "lodging_budget_share": ctx.get("lodging_budget_share"),
            "budget_currency": ctx.get("budget_currency"),
            "budget_per_night_cap": ctx.get("budget_per_night_cap"),
            "search_currency": ctx.get("currency"),
            "serpapi_max_price": params.get("max_price"),
            "local_hard_filter": True,
        },
        "candidates": filtered_candidates,
        "pagination": data.get("serpapi_pagination"),
    }

def build_lodging_queries(ctx: Dict[str, Any]) -> List[str]:
    queries = [ctx["query"]]

    destination = ctx["destination"]
    lodging_type = ctx.get("lodging_type") or "飯店"

    for area in ctx.get("preferred_areas", [])[:2]:
        query = _build_lodging_query(destination, _compact(area), lodging_type)
        if query and query not in queries:
            queries.append(query)

    return queries[:1]

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    input_path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else base_dir / "traced_outputs" / "北京上海測資_result.json"
    with open(input_path, "r", encoding="utf-8") as f:
        context = json.load(f)
    ctx = build_lodging_search_context(context)
    print("="*20 + "看一下搜尋詞" + "="*20)
    print(ctx)
    print("="*20 + "結束" + "="*20)
    print("\n")

    lodging_result = search_lodging_serpapi(ctx, max_results=20)
    print("="*20 + "看一下結果" + "="*20)
    print(lodging_result)
    print("="*20 + "結束" + "="*20)

    # append_text = "hotel_info.json"

    filename = f"{input_path.stem}_hotel_info.json"
    output_path = base_dir / "traced_outputs"
    # save_login_context_output(lodging_result , output_path , filename)
    save_json_output(lodging_result , output_path , filename , "lodging_search")
    # print("="*10 + "開始測試" + "="*10)
    # print(context["trip_context"])
