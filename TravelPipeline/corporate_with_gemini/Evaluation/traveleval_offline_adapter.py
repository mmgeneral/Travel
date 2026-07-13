from __future__ import annotations

import csv
import json
import math
import re
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


EVALUATION_DIR = Path(__file__).resolve().parent
PROJECT_DIR = EVALUATION_DIR.parent
TRAVELEVAL_DIR = Path("/private/tmp/TravelEval")
TRAVELEVAL_DB = TRAVELEVAL_DIR / "environment" / "database"
TRAVELEVAL_CATEGORIES = TRAVELEVAL_DB / "categories.json"

# 改這兩個檔名，就可以直接用 VS Code 的「執行 Python 檔案」測不同資料。
DEFAULT_FINAL_FILENAME = "北京上海測資_result_candidate_llm_input_final_itinerary.json"
DEFAULT_PLANNER_FILENAME = "北京上海測資_result.json"
DEFAULT_OUTPUT_PREFIX = None
DEFAULT_BUDGET_CURRENCY = "CNY"
DEFAULT_OUTPUT_CURRENCY = "CNY"
TWD_PER_CNY = 5.0
CONVERT_COSTS_TO_BUDGET_CURRENCY = True
DEFAULT_MEAL_COST_PER_PERSON_CNY = 80.0
DEFAULT_INTRACITY_TRANSPORT_COST_PER_LEG_CNY = 8.0

CITY_TO_PINYIN = {
    "上海": "shanghai",
    "北京": "beijing",
    "杭州": "hangzhou",
    "南京": "nanjing",
    "蘇州": "suzhou",
    "苏州": "suzhou",
    "重慶": "chongqing",
    "重庆": "chongqing",
    "成都": "chengdu",
    "深圳": "shenzhen",
    "廣州": "guangzhou",
    "广州": "guangzhou",
    "武漢": "wuhan",
    "武汉": "wuhan",
}

TRADITIONAL_TO_SIMPLIFIED = str.maketrans({
    "臺": "台",
    "灣": "湾",
    "滬": "沪",
    "滩": "滩",
    "灘": "滩",
    "園": "园",
    "廟": "庙",
    "觀": "观",
    "館": "馆",
    "樓": "楼",
    "樂": "乐",
    "龍": "龙",
    "湯": "汤",
    "幫": "帮",
    "飯": "饭",
    "橋": "桥",
    "萬": "万",
    "雙": "双",
    "門": "门",
    "區": "区",
    "號": "号",
    "與": "与",
    "內": "内",
    "遊": "游",
    "覽": "览",
    "準": "准",
    "備": "备",
})


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(data: Dict[str, Any], output_dir: Path, filename: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def compact(value: Any) -> str:
    return " ".join(str(value or "").split())


def as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace(",", ""))
    except ValueError:
        match = re.search(r"\d+(?:\.\d+)?", str(value).replace(",", ""))
        if match:
            return float(match.group())
        return default


def normalize_currency(value: Any, default: str = DEFAULT_OUTPUT_CURRENCY) -> str:
    text = compact(value).upper()
    if text in {"RMB", "人民幣", "人民币", "CNY"}:
        return "CNY"
    if text in {"NTD", "TWD", "新台幣", "新臺幣", "台幣", "臺幣"}:
        return "TWD"
    return text or default


def convert_money(amount: Any, from_currency: Any, to_currency: str = DEFAULT_OUTPUT_CURRENCY) -> float:
    value = as_float(amount)
    source = normalize_currency(from_currency)
    target = normalize_currency(to_currency)
    if value == 0 or source == target:
        return round(value, 2)
    if not CONVERT_COSTS_TO_BUDGET_CURRENCY:
        return round(value, 2)
    if source == "TWD" and target == "CNY":
        return round(value / TWD_PER_CNY, 2)
    if source == "CNY" and target == "TWD":
        return round(value * TWD_PER_CNY, 2)
    return round(value, 2)


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", compact(value)).translate(TRADITIONAL_TO_SIMPLIFIED)
    return re.sub(r"[\s（）()\[\]【】,，。:：;；·・\-_/]", "", text.lower())


def parse_minutes(value: Any) -> Optional[int]:
    text = compact(value)
    match = re.match(r"^(\d{1,2}):(\d{2})$", text)
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2))
    if hour == 24 and minute == 0:
        return 24 * 60
    if hour > 24 or minute > 59:
        return None
    return hour * 60 + minute


def duration_hours(start: Any, end: Any) -> float:
    start_m = parse_minutes(start)
    end_m = parse_minutes(end)
    if start_m is None or end_m is None:
        return 0.0
    if end_m < start_m:
        end_m += 24 * 60
    return max(0.0, (end_m - start_m) / 60)


def average(scores: List[Optional[float]]) -> Optional[float]:
    values = [float(score) for score in scores if isinstance(score, (int, float))]
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def clamp01(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 4)


def planner_terms(planner: Dict[str, Any]) -> Dict[str, Any]:
    brief = planner.get("planning_brief_5w1h") or {}
    trip = planner.get("trip_context") or {}
    who = brief.get("who") or {}
    where = brief.get("where") or {}
    when = brief.get("when") or {}
    what = brief.get("what") or {}
    how = brief.get("how") or {}

    days = int(when.get("duration_days") or len(planner.get("itinerary") or []) or 1)
    nights = int(when.get("duration_nights") or max(0, days - 1))

    raw_budget = how.get("budget")
    if isinstance(raw_budget, dict):
        budget_amount = raw_budget.get("amount")
        budget_currency = raw_budget.get("currency")
    else:
        budget_amount = raw_budget
        budget_currency = how.get("budget_currency")

    return {
        "origin": compact(where.get("origin") or trip.get("origin")),
        "destination": compact(where.get("destination") or trip.get("destination")),
        "days": days,
        "nights": nights,
        "travelers": int(who.get("traveler_count") or 1),
        "budget": as_float(budget_amount, 0.0),
        "budget_currency": normalize_currency(budget_currency, DEFAULT_BUDGET_CURRENCY),
        "preferred_transportation": compact(how.get("preferred_transportation")),
        "pace": compact(how.get("pace") or "moderate"),
        "key_areas": [compact(item) for item in where.get("key_areas") or [] if compact(item)],
        "avoid_areas": [compact(item) for item in where.get("avoid_areas") or [] if compact(item)],
        "lodging_preferences": [compact(item) for item in what.get("lodging_preferences") or [] if compact(item)],
        "food_preferences": [compact(item) for item in what.get("food_preferences") or [] if compact(item)],
        "desired_experiences": [compact(item) for item in what.get("desired_experiences") or [] if compact(item)],
        "activity_types": [compact(item) for item in what.get("activity_types") or [] if compact(item)],
        "arrival_anchor": (trip.get("arrival_anchor") or {}).get("name"),
        "return_anchor": (trip.get("return_anchor") or {}).get("name"),
        "trip_goal": compact((brief.get("why") or {}).get("trip_goal")),
    }


def selected_hotel(final_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    lodgings = final_result.get("selected_lodging_by_night") or []
    if not lodgings or not isinstance(lodgings[0], dict):
        return None
    hotel = lodgings[0].get("hotel")
    return hotel if isinstance(hotel, dict) else None


def candidate_evidence(final_result: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    evidence = {}
    for decision in final_result.get("slot_decisions") or []:
        candidate = decision.get("selected_candidate") or {}
        slot_id = decision.get("slot_id")
        if slot_id and candidate:
            evidence[slot_id] = candidate
    return evidence


def activity_location(slot: Dict[str, Any], hotel_name: str) -> str:
    slot_type = compact(slot.get("slot_type"))
    if slot_type == "lodging" and hotel_name:
        return hotel_name
    return compact(slot.get("place_query") or slot.get("route_exit_place") or slot.get("activity"))


def travel_eval_activity_type(slot: Dict[str, Any]) -> str:
    slot_type = compact(slot.get("slot_type"))
    text = compact(slot.get("activity"))
    if slot_type == "transport" or slot.get("is_commute"):
        return "intracity_transport"
    if slot_type == "meal":
        return "meal"
    if slot_type == "lodging":
        if "入住" in text:
            return "accommodation_check_in"
        if "返程" in text or "返回" in text:
            return "accommodation_check_out"
        return "accommodation"
    if slot_type == "attraction":
        return "attraction"
    return slot_type or "other"


def normalize_slot_cost(slot: Dict[str, Any], travelers: int, output_currency: str) -> Tuple[float, Optional[Dict[str, Any]]]:
    raw = slot.get("cost_estimate")
    if not isinstance(raw, dict):
        return 0.0, None

    source_currency = normalize_currency(raw.get("currency"), output_currency)
    total = raw.get("party_total")
    if total is None:
        total = raw.get("total")
    if total is None:
        total = raw.get("amount")
    if total is None and raw.get("per_person") is not None:
        total = as_float(raw.get("per_person")) * max(1, travelers)

    converted_total = convert_money(total, source_currency, output_currency)
    return converted_total, {
        "original": raw,
        "converted_total": converted_total,
        "converted_currency": output_currency,
        "source_currency": source_currency,
        "conversion": f"{output_currency}=TWD/{TWD_PER_CNY:g}" if source_currency == "TWD" and output_currency == "CNY" else None,
    }


def fallback_slot_cost(
    slot: Dict[str, Any],
    activity_type: str,
    location_name: str,
    travelers: int,
    output_currency: str,
    sandbox: Dict[str, Any],
) -> Tuple[float, Optional[Dict[str, Any]]]:
    activity = {"type": activity_type, "location_name": location_name}

    if activity_type == "intracity_transport":
        total = convert_money(DEFAULT_INTRACITY_TRANSPORT_COST_PER_LEG_CNY, "CNY", output_currency)
        return total, {
            "original": {
                "currency": "CNY",
                "party_total": DEFAULT_INTRACITY_TRANSPORT_COST_PER_LEG_CNY,
                "source": "fallback_intracity_transport_default",
                "confidence": 0.45,
                "note": "無 slot cost_estimate 時，市內交通每段以整團 8 CNY 粗估。",
            },
            "converted_total": total,
            "converted_currency": output_currency,
            "source_currency": "CNY",
            "fallback": True,
        }

    if activity_type in {"accommodation", "accommodation_check_in", "accommodation_check_out"}:
        return 0.0, {
            "original": {
                "currency": output_currency,
                "party_total": 0,
                "source": "lodging_cost_counted_in_accommodation_section",
                "confidence": 1.0,
                "note": "住宿總價已在 accommodation.room_type 計算，schedule slot 不重複計費。",
            },
            "converted_total": 0.0,
            "converted_currency": output_currency,
            "source_currency": output_currency,
            "fallback": True,
        }

    row, dataset, match_score = find_sandbox_match(activity, sandbox)
    if row and row.get("price") not in {None, ""}:
        per_person = as_float(row.get("price"))
        total = convert_money(per_person * max(1, travelers), "CNY", output_currency)
        return total, {
            "original": {
                "currency": "CNY",
                "per_person": per_person,
                "party_total": round(per_person * max(1, travelers), 2),
                "source": f"traveleval_{dataset}",
                "confidence": 0.85 if match_score >= 0.8 else 0.65,
                "note": f"Matched {row.get('name')} from TravelEval {dataset}.",
            },
            "converted_total": total,
            "converted_currency": output_currency,
            "source_currency": "CNY",
            "matched_name": row.get("name"),
            "match_score": match_score,
            "fallback": True,
        }

    text = normalize_text(" ".join([location_name, slot.get("activity") or ""]))
    looks_like_food = any(term in text for term in ["餐", "美食", "咖啡", "汤包", "小笼包", "本帮菜"])
    if activity_type == "meal" or looks_like_food:
        total = convert_money(DEFAULT_MEAL_COST_PER_PERSON_CNY * max(1, travelers), "CNY", output_currency)
        return total, {
            "original": {
                "currency": "CNY",
                "per_person": DEFAULT_MEAL_COST_PER_PERSON_CNY,
                "party_total": DEFAULT_MEAL_COST_PER_PERSON_CNY * max(1, travelers),
                "source": "fallback_meal_default",
                "confidence": 0.4,
                "note": "無 TravelEval restaurant 精準匹配時，每人餐費以 80 CNY 粗估。",
            },
            "converted_total": total,
            "converted_currency": output_currency,
            "source_currency": "CNY",
            "fallback": True,
        }

    if activity_type == "attraction":
        return 0.0, {
            "original": {
                "currency": output_currency,
                "party_total": 0,
                "source": "fallback_unknown_attraction_free_or_missing",
                "confidence": 0.3,
                "note": "景點未匹配票價，暫以 0 計；應由 FinalItineraryAgent cost_estimate 或票價資料補齊。",
            },
            "converted_total": 0.0,
            "converted_currency": output_currency,
            "source_currency": output_currency,
            "fallback": True,
        }

    return 0.0, None


def build_user_query(planner: Dict[str, Any]) -> Dict[str, Any]:
    terms = planner_terms(planner)
    transport_preferences = []
    if terms["preferred_transportation"]:
        transport_preferences = re.split(r"[與和、,/，\s]+", terms["preferred_transportation"])
        transport_preferences = [item for item in transport_preferences if item]

    return {
        "uid": "corporate_with_gemini_offline_case",
        "start_city": terms["origin"],
        "target_city": terms["destination"],
        "days": terms["days"],
        "people_number": terms["travelers"],
        "budget": terms["budget"],
        "budget_currency": terms["budget_currency"],
        "transportation": {"preferences": transport_preferences, "constraints": []},
        "accommodations": {"preferences": terms["lodging_preferences"], "constraints": []},
        "diet": {"preferences": terms["food_preferences"], "constraints": []},
        "attractions": {
            "preferences": terms["key_areas"] + terms["desired_experiences"] + terms["activity_types"],
            "constraints": terms["avoid_areas"],
        },
        "rhythm": {"preferences": [terms["pace"]]},
        "nature_language": terms["trip_goal"],
    }


def estimate_cost_breakdown(ai_plan: Dict[str, Any]) -> Dict[str, float]:
    costs = {
        "attractions": 0.0,
        "intercity_transportation": 0.0,
        "intracity_transportation": 0.0,
        "accommodation": 0.0,
        "meals": 0.0,
        "other": 0.0,
    }

    for transport in (ai_plan.get("intercity_transport") or {}).get("transport_type") or []:
        details = transport.get("details") or {}
        costs["intercity_transportation"] += as_float(details.get("price")) * as_float(details.get("number"), 1.0)

    for room in (ai_plan.get("accommodation") or {}).get("room_type") or []:
        costs["accommodation"] += (
            as_float(room.get("quantity"), 1.0)
            * as_float(room.get("price_per_night"))
            * as_float(room.get("nights"), 1.0)
        )

    for day in ai_plan.get("daily_plans") or []:
        for activity in day.get("activities") or []:
            cost = as_float(activity.get("cost"))
            activity_type = activity.get("type")
            if activity_type == "attraction":
                costs["attractions"] += cost
            elif activity_type == "meal":
                costs["meals"] += cost
            elif activity_type not in {"intracity_transport", "accommodation", "accommodation_check_in", "accommodation_check_out"}:
                costs["other"] += cost
            costs["intracity_transportation"] += as_float(activity.get("transportation_cost"))
        costs["intracity_transportation"] += as_float((day.get("ending_point") or {}).get("transportation_cost"))

    costs["total"] = round(sum(costs.values()), 2)
    return {key: round(value, 2) for key, value in costs.items()}


def convert_to_traveleval_plan(final_result: Dict[str, Any], planner: Dict[str, Any]) -> Dict[str, Any]:
    terms = planner_terms(planner)
    sandbox = load_sandbox(terms["destination"])
    hotel = selected_hotel(final_result) or {}
    hotel_name = compact(hotel.get("name") or "住宿")
    hotel_price = hotel.get("price") if isinstance(hotel.get("price"), dict) else {}
    output_currency = terms["budget_currency"] or DEFAULT_OUTPUT_CURRENCY
    hotel_currency = normalize_currency(hotel_price.get("currency"), "TWD")
    nights = int(hotel_price.get("nights") or terms["nights"] or 0)
    room_quantity = max(1, math.ceil(terms["travelers"] / 2))
    hotel_total_raw = hotel_price.get("total")
    if hotel_total_raw is None:
        hotel_total_raw = hotel_price.get("party_total")
    if hotel_total_raw is None:
        hotel_total_raw = hotel_price.get("amount")
    if hotel_total_raw is None and hotel_price.get("per_person") is not None:
        hotel_total_raw = as_float(hotel_price.get("per_person")) * max(1, terms["travelers"])

    hotel_total = convert_money(hotel_total_raw, hotel_currency, output_currency)
    if hotel_total > 0 and nights > 0:
        per_night = round(hotel_total / (nights * room_quantity), 2)
    else:
        per_night_raw = hotel_price.get("per_night")
        if per_night_raw is None:
            per_night_raw = hotel_price.get("per_person")
        per_night = convert_money(per_night_raw, hotel_currency, output_currency)

    daily_plans = []
    for day in final_result.get("final_itinerary") or []:
        activities = []
        for slot in day.get("schedule") or []:
            start_time = compact(slot.get("start_time"))
            end_time = compact(slot.get("end_time"))
            activity_type = travel_eval_activity_type(slot)
            location_name = activity_location(slot, hotel_name)
            slot_cost, cost_estimate = normalize_slot_cost(slot, terms["travelers"], output_currency)
            if cost_estimate is None:
                slot_cost, cost_estimate = fallback_slot_cost(
                    slot,
                    activity_type,
                    location_name,
                    terms["travelers"],
                    output_currency,
                    sandbox,
                )
            activities.append({
                "type": activity_type,
                "location_name": location_name,
                "description": compact(slot.get("activity")),
                "start_time": start_time,
                "end_time": end_time,
                "cost": 0 if activity_type == "intracity_transport" else slot_cost,
                "transportation_to": terms["preferred_transportation"],
                "transportation_cost": slot_cost if activity_type == "intracity_transport" else 0,
                "details": None,
                "cost_estimate": cost_estimate,
                "source_slot_id": slot.get("slot_id"),
                "source_slot_type": slot.get("slot_type"),
            })

        starting_point = hotel_name or compact(terms["arrival_anchor"] or terms["destination"])
        ending_name = hotel_name
        if day.get("day") == terms["days"]:
            ending_name = compact(terms["return_anchor"] or ending_name)
        if activities and activities[-1].get("location_name"):
            ending_name = activities[-1]["location_name"]

        daily_plans.append({
            "day": day.get("day"),
            "date_description": day.get("date_description"),
            "starting_point": starting_point,
            "activities": activities,
            "ending_point": {
                "type": "ending_point",
                "location_name": ending_name,
                "description": "當日結束點",
                "start_time": activities[-1]["end_time"] if activities else "22:00",
                "end_time": activities[-1]["end_time"] if activities else "22:00",
                "cost": 0,
                "transportation_to": terms["preferred_transportation"],
                "transportation_cost": 0,
                "details": None,
            },
        })

    ai_plan = {
        "summary": {
            "departure": terms["origin"],
            "destination": terms["destination"],
            "total_days": terms["days"],
            "total_travelers": terms["travelers"],
            "total_budget": terms["budget"],
            "budget_currency": output_currency,
            "cost_currency": output_currency,
            "currency_conversion": {
                "enabled": CONVERT_COSTS_TO_BUDGET_CURRENCY,
                "TWD_PER_CNY": TWD_PER_CNY,
                "formula": "CNY = TWD / TWD_PER_CNY",
            },
            "calculated_total_cost": 0,
            "is_within_budget": False,
            "source_note": "Converted from FinalItineraryAgent output for no-Gaode offline scoring.",
        },
        "intercity_transport": {
            "transport_type": [],
            "note": "FinalItineraryAgent output has arrival/return anchors but no train or flight ticket detail.",
        },
        "accommodation": {
            "hotel_name": hotel_name,
            "room_type": [{
                "type": "雙人房",
                "quantity": room_quantity,
                "price_per_night": per_night,
                "nights": nights,
                "currency": output_currency,
                "original_price": hotel_price,
                "source": hotel_price.get("source"),
                "availability_confirmed": hotel_price.get("availability_confirmed"),
            }],
            "total_cost": hotel_total or round(per_night * nights * room_quantity, 2),
        },
        "daily_plans": daily_plans,
        "cost_breakdown": {},
        "conversion_notes": [
            "保持 TravelEval 期待的 summary/intercity_transport/accommodation/daily_plans/cost_breakdown 結構。",
            (
                f"預算以 {output_currency} 評分；SerpApi 住宿若是 TWD，先用 CNY = TWD / {TWD_PER_CNY:g} 換算。"
                if CONVERT_COSTS_TO_BUDGET_CURRENCY
                else "已關閉幣別換算；成本會保留來源數字，budget score 不適合和 paper 直接比較。"
            ),
            "若 final result 的 slot 有 cost_estimate 會優先採用；缺少時用 TravelEval DB 或保守預設值 fallback。",
        ],
    }

    cost_breakdown = estimate_cost_breakdown(ai_plan)
    ai_plan["cost_breakdown"] = cost_breakdown
    ai_plan["summary"]["calculated_total_cost"] = cost_breakdown["total"]
    budget = terms["budget"]
    ai_plan["summary"]["is_within_budget"] = bool(budget <= 0 or cost_breakdown["total"] <= budget)
    return ai_plan


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def load_sandbox(destination: str) -> Dict[str, Any]:
    city = CITY_TO_PINYIN.get(destination, normalize_text(destination))
    return {
        "city_key": city,
        "attractions": read_csv_rows(TRAVELEVAL_DB / "attractions" / city / "attractions.csv"),
        "restaurants": read_csv_rows(TRAVELEVAL_DB / "restaurants" / city / f"restaurants_{city}.csv"),
        "accommodations": read_csv_rows(TRAVELEVAL_DB / "accommodations" / city / "accommodations.csv"),
        "source": str(TRAVELEVAL_DB),
    }


def best_match(name: str, rows: List[Dict[str, Any]], threshold: float = 0.58) -> Tuple[Optional[Dict[str, Any]], float]:
    norm_name = normalize_text(name)
    if not norm_name:
        return None, 0.0

    best_row = None
    best_score = 0.0
    for row in rows:
        row_name = normalize_text(row.get("name"))
        if not row_name:
            continue
        if norm_name == row_name:
            score = 1.0
        elif norm_name in row_name or row_name in norm_name:
            score = max(0.82, min(len(norm_name), len(row_name)) / max(len(norm_name), len(row_name)))
        elif "上海中心" in norm_name and "上海中心" in row_name:
            score = 0.86
        elif "迪士尼" in norm_name and "迪士尼" in row_name:
            score = 0.86
        else:
            score = SequenceMatcher(None, norm_name, row_name).ratio()
        if score > best_score:
            best_row = row
            best_score = score

    if best_score < threshold:
        return None, round(best_score, 4)
    return best_row, round(best_score, 4)


def find_sandbox_match(activity: Dict[str, Any], sandbox: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str, float]:
    name = compact(activity.get("location_name"))
    activity_type = activity.get("type")
    if activity_type == "meal":
        row, score = best_match(name, sandbox.get("restaurants") or [])
        return row, "restaurants", score
    if activity_type in {"accommodation", "accommodation_check_in", "accommodation_check_out"}:
        row, score = best_match(name, sandbox.get("accommodations") or [])
        return row, "accommodations", score
    if activity_type == "attraction":
        row, score = best_match(name, sandbox.get("attractions") or [])
        return row, "attractions", score
    return None, "not_checked", 0.0


def has_candidate_coordinates(activity: Dict[str, Any], evidence: Dict[str, Dict[str, Any]]) -> bool:
    candidate = evidence.get(activity.get("source_slot_id")) or {}
    return candidate.get("latitude") is not None and candidate.get("longitude") is not None


def score_place_grounding(ai_plan: Dict[str, Any], sandbox: Dict[str, Any], evidence: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    checked = []
    for day in ai_plan.get("daily_plans") or []:
        for activity in day.get("activities") or []:
            if activity.get("type") == "intracity_transport":
                continue
            name = compact(activity.get("location_name"))
            if not name:
                continue
            row, dataset, match_score = find_sandbox_match(activity, sandbox)
            grounded_by_candidate = has_candidate_coordinates(activity, evidence)
            checked.append({
                "name": name,
                "type": activity.get("type"),
                "grounded": bool(row or grounded_by_candidate),
                "matched_dataset": dataset if row else None,
                "matched_name": row.get("name") if row else None,
                "match_score": match_score,
                "candidate_coordinates": grounded_by_candidate,
            })

    if not checked:
        return {"score": None, "value": None, "checked": checked}
    ungrounded = [item for item in checked if not item["grounded"]]
    rate = len(ungrounded) / len(checked)
    return {"score": clamp01(1 - rate), "value": round(rate, 4), "checked": checked}


def within_opening_hours(activity: Dict[str, Any], row: Dict[str, Any]) -> bool:
    start = parse_minutes(activity.get("start_time"))
    end = parse_minutes(activity.get("end_time"))
    open_time = parse_minutes(row.get("opentime"))
    close_time = parse_minutes(row.get("endtime"))
    if None in {start, end, open_time, close_time}:
        return True
    if end < start:
        end += 24 * 60
    if close_time < open_time:
        return start >= open_time or end <= close_time
    return start >= open_time and end <= close_time


def score_opening_hours(ai_plan: Dict[str, Any], sandbox: Dict[str, Any]) -> Dict[str, Any]:
    checked = []
    for day in ai_plan.get("daily_plans") or []:
        for activity in day.get("activities") or []:
            if activity.get("type") != "attraction":
                continue
            row, _, match_score = find_sandbox_match(activity, sandbox)
            if not row:
                continue
            ok = within_opening_hours(activity, row)
            checked.append({
                "name": activity.get("location_name"),
                "matched_name": row.get("name"),
                "visit_time": f"{activity.get('start_time')}~{activity.get('end_time')}",
                "opening_time": f"{row.get('opentime')}~{row.get('endtime')}",
                "match_score": match_score,
                "ok": ok,
            })
    if not checked:
        return {"score": None, "value": None, "checked": checked}
    violation_rate = sum(1 for item in checked if not item["ok"]) / len(checked)
    return {"score": clamp01(1 - violation_rate), "value": round(violation_rate, 4), "checked": checked}


def score_visit_duration(ai_plan: Dict[str, Any], sandbox: Dict[str, Any]) -> Dict[str, Any]:
    checked = []
    for day in ai_plan.get("daily_plans") or []:
        for activity in day.get("activities") or []:
            if activity.get("type") != "attraction":
                continue
            row, _, match_score = find_sandbox_match(activity, sandbox)
            if not row:
                continue
            planned = duration_hours(activity.get("start_time"), activity.get("end_time"))
            min_time = as_float(row.get("recommendmintime"))
            max_time = as_float(row.get("recommendmaxtime"))
            if planned <= 0 or min_time <= 0 or max_time <= 0:
                score = None
            elif planned < min_time:
                score = clamp01(planned / min_time)
            elif planned > max_time:
                score = clamp01(max_time / planned)
            else:
                score = 1.0
            checked.append({
                "name": activity.get("location_name"),
                "matched_name": row.get("name"),
                "planned_hours": round(planned, 2),
                "recommended_hours": [min_time, max_time],
                "match_score": match_score,
                "score": score,
            })
    return {"score": average([item["score"] for item in checked]), "checked": checked}


def full_text(final_result: Dict[str, Any]) -> str:
    return json.dumps(final_result.get("final_itinerary") or [], ensure_ascii=False)


def term_coverage(terms: List[str], text: str) -> Dict[str, Any]:
    normalized_text = normalize_text(text)
    matched = []
    missing = []
    for term in terms:
        norm = normalize_text(term)
        synonyms = [norm]
        if norm == normalize_text("小籠包"):
            synonyms.append(normalize_text("湯包"))
        if norm and any(item in normalized_text for item in synonyms):
            matched.append(term)
        else:
            missing.append(term)
    score = 1.0 if not terms else len(matched) / len(terms)
    return {"score": round(score, 4), "matched": matched, "missing": missing}


def desired_experience_coverage(terms: List[str], ai_plan: Dict[str, Any], text: str) -> Dict[str, Any]:
    normalized_text = normalize_text(text)
    activity_types = [
        activity.get("type")
        for day in ai_plan.get("daily_plans") or []
        for activity in day.get("activities") or []
    ]
    matched = []
    missing = []

    for term in terms:
        norm = normalize_text(term)
        is_food_need = any(word in norm for word in ["美食", "餐", "小吃"])
        is_sightseeing_need = any(word in norm for word in ["景点", "景點", "游览", "遊覽", "观光", "觀光"])
        if norm and norm in normalized_text:
            matched.append(term)
        elif is_food_need and "meal" in activity_types:
            matched.append(term)
        elif is_sightseeing_need and "attraction" in activity_types:
            matched.append(term)
        else:
            missing.append(term)

    score = 1.0 if not terms else len(matched) / len(terms)
    return {"score": round(score, 4), "matched": matched, "missing": missing}


def score_schedule_validity(ai_plan: Dict[str, Any]) -> Dict[str, Any]:
    issues = []
    checked_days = 0
    for day in ai_plan.get("daily_plans") or []:
        previous_end = None
        checked_days += 1
        for activity in day.get("activities") or []:
            start = parse_minutes(activity.get("start_time"))
            end = parse_minutes(activity.get("end_time"))
            if start is None or end is None:
                issues.append({"day": day.get("day"), "activity": activity.get("description"), "issue": "missing_or_bad_time"})
                continue
            if end < start:
                end += 24 * 60
            if end <= start:
                issues.append({"day": day.get("day"), "activity": activity.get("description"), "issue": "non_positive_duration"})
            if previous_end is not None and start < previous_end:
                issues.append({"day": day.get("day"), "activity": activity.get("description"), "issue": "overlap"})
            previous_end = end
    if checked_days == 0:
        return {"score": None, "issues": issues}
    return {"score": clamp01(1 - len(issues) / max(1, checked_days)), "issues": issues}


def score_daily_utilization(ai_plan: Dict[str, Any]) -> Dict[str, Any]:
    day_scores = []
    for day in ai_plan.get("daily_plans") or []:
        active_hours = 0.0
        for activity in day.get("activities") or []:
            if activity.get("type") in {"attraction", "meal"}:
                active_hours += duration_hours(activity.get("start_time"), activity.get("end_time"))
        score = clamp01(active_hours / 8.0)
        day_scores.append({"day": day.get("day"), "active_hours": round(active_hours, 2), "score": score})
    return {"score": average([item["score"] for item in day_scores]), "days": day_scores}


def score_pace(ai_plan: Dict[str, Any], pace: str) -> Dict[str, Any]:
    target = 2.5
    if pace == "slow":
        target = 1.5
    elif pace in {"fast", "packed"}:
        target = 3.5
    day_scores = []
    for day in ai_plan.get("daily_plans") or []:
        count = sum(1 for item in day.get("activities") or [] if item.get("type") == "attraction")
        score = clamp01(1 - abs(count - target) / target)
        day_scores.append({"day": day.get("day"), "attraction_count": count, "target": target, "score": score})
    return {"score": average([item["score"] for item in day_scores]), "days": day_scores}


def score_accommodation(ai_plan: Dict[str, Any], sandbox: Dict[str, Any], expected_nights: int) -> Dict[str, Any]:
    hotel_name = compact((ai_plan.get("accommodation") or {}).get("hotel_name"))
    rooms = (ai_plan.get("accommodation") or {}).get("room_type") or []
    nights = sum(int(as_float(room.get("nights"))) for room in rooms)
    row, match_score = best_match(hotel_name, sandbox.get("accommodations") or [])
    nights_score = 1.0 if expected_nights == 0 or nights == expected_nights else 0.0
    grounding_score = 1.0 if row else 0.5 if hotel_name else 0.0
    return {
        "score": average([nights_score, grounding_score]),
        "hotel_name": hotel_name,
        "expected_nights": expected_nights,
        "planned_nights": nights,
        "matched_name": row.get("name") if row else None,
        "match_score": match_score,
    }


def score_transport_preference(ai_plan: Dict[str, Any], preferred_transportation: str, final_result: Dict[str, Any]) -> Dict[str, Any]:
    if not preferred_transportation:
        return {"score": 1.0, "note": "使用者沒有指定交通偏好"}
    text = normalize_text(full_text(final_result))
    preferred_parts = [normalize_text(part) for part in re.split(r"[與和、,/，\s]+", preferred_transportation) if part]
    matched = [part for part in preferred_parts if part and part in text]
    has_commute_slots = any(
        activity.get("type") == "intracity_transport"
        for day in ai_plan.get("daily_plans") or []
        for activity in day.get("activities") or []
    )
    if len(matched) == len(preferred_parts):
        score = 1.0
    elif has_commute_slots:
        score = 0.65
    else:
        score = 0.35
    return {"score": score, "preferred": preferred_transportation, "matched_terms": matched, "has_commute_slots": has_commute_slots}


def score_cost_completeness(cost_breakdown: Dict[str, float]) -> Dict[str, Any]:
    expected = ["accommodation", "meals", "intracity_transportation", "intercity_transportation", "attractions"]
    present = [key for key in expected if as_float(cost_breakdown.get(key)) > 0]
    return {"score": round(len(present) / len(expected), 4), "present_categories": present, "expected_categories": expected}


def score_budget_efficiency(total_cost: float, budget: float) -> Dict[str, Any]:
    if budget <= 0 or total_cost <= 0:
        return {"score": None, "note": "缺少預算或總成本，無法計算"}
    return {"score": clamp01(budget / total_cost), "budget": budget, "total_cost": total_cost}


def score_diversity(ai_plan: Dict[str, Any]) -> Dict[str, Any]:
    categories = set()
    for day in ai_plan.get("daily_plans") or []:
        for activity in day.get("activities") or []:
            text = normalize_text(activity.get("description") or activity.get("location_name"))
            if activity.get("type") == "meal":
                categories.add("food")
            if "迪士尼" in text:
                categories.add("theme_park")
            if any(term in text for term in ["外滩", "中心", "地标"]):
                categories.add("landmark")
            if any(term in text for term in ["豫园", "城隍庙", "文化"]):
                categories.add("culture")
            if any(term in text for term in ["咖啡", "法租界"]):
                categories.add("leisure")
    return {"score": clamp01(len(categories) / 4), "categories": sorted(categories)}


def metric(score: Optional[float], value: Any = None, note: str = "", details: Any = None) -> Dict[str, Any]:
    return {"score": score, "value": value, "note": note, "details": details}


def iter_activities(ai_plan: Dict[str, Any], activity_type: Optional[str] = None) -> List[Dict[str, Any]]:
    output = []
    for day in ai_plan.get("daily_plans") or []:
        for activity in day.get("activities") or []:
            if activity_type is None or activity.get("type") == activity_type:
                output.append(activity)
    return output


def parse_types(value: Any) -> List[str]:
    text = compact(value).strip("{}")
    return [compact(item) for item in text.split(";") if compact(item)]


def load_categories() -> Dict[str, Any]:
    if not TRAVELEVAL_CATEGORIES.exists():
        return {}
    return json.loads(TRAVELEVAL_CATEGORIES.read_text(encoding="utf-8"))


def attraction_records(ai_plan: Dict[str, Any], sandbox: Dict[str, Any]) -> List[Dict[str, Any]]:
    records = []
    for activity in iter_activities(ai_plan, "attraction"):
        row, _, match_score = find_sandbox_match(activity, sandbox)
        if not row:
            continue
        records.append({
            "activity": activity,
            "row": row,
            "name": activity.get("location_name"),
            "matched_name": row.get("name"),
            "duration_hours": duration_hours(activity.get("start_time"), activity.get("end_time")),
            "star": as_float(row.get("star"), 3.0),
            "types": parse_types(row.get("type")),
            "match_score": match_score,
        })
    return records


def preference_match_score(record: Dict[str, Any], user_query: Dict[str, Any]) -> float:
    prefs = user_query.get("attractions", {}).get("preferences", []) or []
    constraints = user_query.get("attractions", {}).get("constraints", []) or []
    haystack = normalize_text(" ".join([record.get("matched_name") or ""] + record.get("types", [])))

    if any(normalize_text(item) and normalize_text(item) in haystack for item in constraints):
        return 0.0

    matched_count = sum(1 for item in prefs if normalize_text(item) and normalize_text(item) in haystack)
    return 1.0 + matched_count


def calculate_edi(records: List[Dict[str, Any]]) -> Optional[float]:
    categories = load_categories().get("attractions") or []
    if not records or not categories:
        return None

    type_scores: Dict[str, float] = {}
    for record in records:
        types = [item for item in record.get("types", []) if item]
        if not types:
            continue
        share = 1 / len(types)
        for item in types:
            type_scores[item] = type_scores.get(item, 0.0) + share

    total = sum(type_scores.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for score in type_scores.values():
        p = score / total
        entropy -= p * math.log2(p)
    return round(entropy / math.log2(len(categories)), 4)


def calculate_time_metrics(ai_plan: Dict[str, Any], visit_duration: Dict[str, Any]) -> Dict[str, Optional[float]]:
    str_value = average([item.get("score") for item in visit_duration.get("checked") or []])

    daily_values = []
    total_attraction_hours = 0.0
    for day in ai_plan.get("daily_plans") or []:
        activities = day.get("activities") or []
        starts = [parse_minutes(item.get("start_time")) for item in activities]
        ends = [parse_minutes(item.get("end_time")) for item in activities]
        starts = [item for item in starts if item is not None]
        ends = [item for item in ends if item is not None]
        span_hours = 0.0
        if starts and ends:
            span_hours = max(0.0, (max(ends) - min(starts)) / 60)
        attraction_hours = sum(
            duration_hours(item.get("start_time"), item.get("end_time"))
            for item in activities
            if item.get("type") == "attraction"
        )
        total_attraction_hours += attraction_hours
        if span_hours > 0:
            daily_values.append(clamp01(attraction_hours / span_hours))

    days = max(1, int((ai_plan.get("summary") or {}).get("total_days") or len(ai_plan.get("daily_plans") or []) or 1))
    return {
        "STR": str_value,
        "DTU": average(daily_values),
        "OTU": round(total_attraction_hours / (24 * days), 4),
    }


def calculate_utility_metrics(ai_plan: Dict[str, Any], user_query: Dict[str, Any], sandbox: Dict[str, Any]) -> Dict[str, Any]:
    records = attraction_records(ai_plan, sandbox)
    attraction_count = len(iter_activities(ai_plan, "attraction"))
    days = max(1, int((ai_plan.get("summary") or {}).get("total_days") or 1))
    target_density = 3.54

    total_weighted_quality_time = 0.0
    total_attraction_time = 0.0
    experience_value = 0.0
    for record in records:
        duration = record["duration_hours"]
        normalized_star = min(1.0, record["star"] / 5)
        total_weighted_quality_time += normalized_star * duration
        total_attraction_time += duration
        experience_value += record["star"] * preference_match_score(record, user_query)

    return {
        "records": records,
        "experience_value": round(experience_value, 4),
        "EDI": calculate_edi(records),
        "ADS": round((attraction_count / days) / target_density, 4) if days else None,
        "AQE": round(total_weighted_quality_time / total_attraction_time, 4) if total_attraction_time > 0 else None,
        "Profit": round(experience_value / len(records), 4) if records else None,
    }


def calculate_fictitious_attraction_rate(ai_plan: Dict[str, Any], sandbox: Dict[str, Any], evidence: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    checked = []
    for activity in iter_activities(ai_plan, "attraction"):
        row, _, match_score = find_sandbox_match(activity, sandbox)
        grounded = bool(row or has_candidate_coordinates(activity, evidence))
        checked.append({
            "name": activity.get("location_name"),
            "grounded": grounded,
            "matched_name": row.get("name") if row else None,
            "match_score": match_score,
            "candidate_coordinates": has_candidate_coordinates(activity, evidence),
        })
    if not checked:
        return {"value": None, "checked": checked}
    fake_count = sum(1 for item in checked if not item["grounded"])
    return {"value": round(fake_count / len(checked), 4), "checked": checked}


def partial_metric_row(
    dimension: str,
    metric: str,
    direction: str,
    value: Any,
    status: str,
    note: str = "",
    details: Any = None,
) -> Dict[str, Any]:
    if isinstance(value, float):
        value = round(value, 4)
    return {
        "dimension": dimension,
        "metric": metric,
        "direction": direction,
        "value": value,
        "status": status,
        "note": note,
        "details": details,
    }


def cost_row_status(value: Any) -> str:
    return "estimated_or_computed" if as_float(value) > 0 else "missing_or_zero"


def build_partial_metrics_table(final_result: Dict[str, Any], planner: Dict[str, Any], ai_plan: Dict[str, Any]) -> Dict[str, Any]:
    terms = planner_terms(planner)
    user_query = build_user_query(planner)
    sandbox = load_sandbox(terms["destination"])
    evidence = candidate_evidence(final_result)
    cost_breakdown = ai_plan.get("cost_breakdown") or {}
    recalculated_cost = estimate_cost_breakdown(ai_plan)
    total_cost = as_float(cost_breakdown.get("total"))
    budget = as_float(user_query.get("budget"))

    far = calculate_fictitious_attraction_rate(ai_plan, sandbox, evidence)
    opening_hours = score_opening_hours(ai_plan, sandbox)
    visit_duration = score_visit_duration(ai_plan, sandbox)
    schedule_validity = score_schedule_validity(ai_plan)
    accommodation = score_accommodation(ai_plan, sandbox, terms["nights"])
    transport = score_transport_preference(ai_plan, terms["preferred_transportation"], final_result)
    food_coverage = term_coverage(terms["food_preferences"], full_text(final_result))
    time_metrics = calculate_time_metrics(ai_plan, visit_duration)
    utility = calculate_utility_metrics(ai_plan, user_query, sandbox)

    planned_people = as_float(ai_plan.get("summary", {}).get("total_travelers"))
    requested_people = as_float(user_query.get("people_number"))
    people_deviation = 0.0 if requested_people <= 0 else abs(planned_people - requested_people) / requested_people
    planned_days = int(ai_plan.get("summary", {}).get("total_days") or 0)
    requested_days = int(user_query.get("days") or 0)
    same_days = planned_days == requested_days
    no_schedule_issues = not schedule_validity.get("issues")
    total_discrepancy = abs(as_float(cost_breakdown.get("total")) - as_float(recalculated_cost.get("total")))
    cost_deviation = 0.0 if as_float(recalculated_cost.get("total")) <= 0 else total_discrepancy / as_float(recalculated_cost.get("total"))

    has_intercity_need = bool(terms["origin"] and terms["destination"] and normalize_text(terms["origin"]) != normalize_text(terms["destination"]))
    has_intercity_details = bool((ai_plan.get("intercity_transport") or {}).get("transport_type"))
    itd_status = "missing_input" if has_intercity_need and not has_intercity_details else "computed"
    itd_note = "final result 沒有火車/航班班次與時間，無法用 TravelEval intercity sandbox 算 ITD。" if itd_status == "missing_input" else ""

    ttcs_checked = visit_duration.get("checked") or []
    ttcs = None
    if ttcs_checked:
        ttcs = 1.0 if all((item.get("score") or 0) >= 1.0 for item in ttcs_checked) else 0.0

    bcs = None
    if budget > 0 and total_cost > 0:
        bcs = 1.0 if total_cost <= budget else 0.0

    pas_a = accommodation.get("score")
    if not terms["lodging_preferences"] and accommodation.get("planned_nights") == accommodation.get("expected_nights"):
        pas_a = 1.0

    be = None
    if total_cost > 0:
        be = round(utility["experience_value"] / total_cost * 1000, 4)

    computed = [
        partial_metric_row("Accuracy", "CCD", "↓", cost_deviation, "computed_from_converted_plan", "成本表由轉換器重算；可抓 arithmetic mismatch，但不是模型自報成本。"),
        partial_metric_row("Accuracy", "FAR", "↓", far["value"], "computed_partial", "只檢查 attraction；可由 TravelEval sandbox 或候選座標佐證。", far["checked"]),
        partial_metric_row("Accuracy", "VROH", "↓", opening_hours["value"], "computed_partial", "只檢查能匹配到 TravelEval attractions CSV 的景點。", opening_hours["checked"]),
        partial_metric_row("Accuracy", "ITD", "↓", None, itd_status, itd_note),
        partial_metric_row("Accuracy", "PD", "↓", round(people_deviation, 4), "computed"),
        partial_metric_row("Compliance", "BCS", "↑", bcs, "computed", f"成本與預算統一以 {terms['budget_currency']} 計。"),
        partial_metric_row("Compliance", "TCS", "↑", 1.0 if same_days and no_schedule_issues else 0.0, "computed_partial", "檢查天數與時間格式/重疊；未檢查實際交通可行時間。"),
        partial_metric_row("Compliance", "HCS", "↑", 1.0 if people_deviation == 0 else 0.0, "computed_partial"),
        partial_metric_row("Compliance", "PAS-A", "↑", pas_a, "computed_partial", "無住宿偏好時，夜數正確即視為滿足。", accommodation),
        partial_metric_row("Compliance", "PAS-T", "↑", transport.get("score"), "proxy_no_gaode", "不查實際路線，只看偏好交通是否出現在行程/通勤 slot。", transport),
        partial_metric_row("Compliance", "PAS-C", "↑", food_coverage["score"], "proxy_text_match", "用飲食偏好文字覆蓋近似。", food_coverage),
        partial_metric_row("Compliance", "TTCS", "↑", ttcs, "computed_partial", "使用 TravelEval 推薦停留時間；未納入排隊時間模型。", ttcs_checked),
        partial_metric_row("Temporality", "STR", "↑", time_metrics["STR"], "proxy_no_queue_model", "用停留時間落在推薦區間的程度近似，未納入排隊時間。"),
        partial_metric_row("Temporality", "DTU", "↑", time_metrics["DTU"], "proxy_no_queue_model", "用每日景點時數 / 當日行程 span 近似。"),
        partial_metric_row("Temporality", "OTU", "↑", time_metrics["OTU"], "proxy_no_queue_model", "用總景點時數 / 24h*天數近似。"),
        partial_metric_row("Economy", "BE", "↑", be, "proxy_estimated_costs", "TravelEval 公式方向：experience value / actual cost * 1000；目前成本含住宿、景點票、餐費與市內交通估算，仍缺城際交通。"),
        partial_metric_row("Economy", "ACD", "-", cost_breakdown.get("accommodation"), cost_row_status(cost_breakdown.get("accommodation")), "住宿成本。"),
        partial_metric_row("Economy", "ATD", "-", cost_breakdown.get("attractions"), cost_row_status(cost_breakdown.get("attractions")), "景點票價優先來自 slot cost_estimate，缺少時用 TravelEval attraction price fallback。"),
        partial_metric_row("Economy", "ETD", "-", cost_breakdown.get("intercity_transportation"), cost_row_status(cost_breakdown.get("intercity_transportation")), "城際交通成本目前缺資料。"),
        partial_metric_row("Economy", "RTD", "-", cost_breakdown.get("intracity_transportation"), cost_row_status(cost_breakdown.get("intracity_transportation")), "市內交通成本優先來自 slot cost_estimate，缺少時用每段整團 8 CNY fallback。"),
        partial_metric_row("Economy", "MED", "-", cost_breakdown.get("meals"), cost_row_status(cost_breakdown.get("meals")), "餐費優先來自 slot cost_estimate，缺少時用 TravelEval restaurant price 或每人 80 CNY fallback。"),
        partial_metric_row("Economy", "OTD", "-", cost_breakdown.get("other"), cost_row_status(cost_breakdown.get("other")), "其他成本。"),
        partial_metric_row("Utility", "EDI", "↑", utility["EDI"], "computed_partial", "用 TravelEval attraction type 做 Shannon entropy。"),
        partial_metric_row("Utility", "ADS", "↑", utility["ADS"], "computed_partial", "平均每日景點數 / 普通旅行 target density 3.54。"),
        partial_metric_row("Utility", "AQE", "↑", utility["AQE"], "computed_partial", "用景點 star/5 加權停留時間近似。"),
        partial_metric_row("Utility", "Profit", "↑", utility["Profit"], "computed_partial", "用 sandbox star 與偏好匹配近似。"),
    ]

    skipped = [
        partial_metric_row("Accuracy", "FRTC", "↓", None, "skipped_gaode_required", "需要高德 transit API 檢查相鄰 POI 公共交通可達性。"),
        partial_metric_row("Spatiality", "SSR", "↓", None, "skipped_gaode_required", "需要高德路網距離計算 route penalty。"),
        partial_metric_row("Spatiality", "CSM-P90", "↓", None, "skipped_gaode_required", "需要 POI 路網距離與跨日空間錯配計算。"),
        partial_metric_row("Spatiality", "CSM-P95", "↓", None, "skipped_gaode_required", "需要 POI 路網距離與跨日空間錯配計算。"),
    ]

    return {
        "title": "TravelEval partial metrics without Gaode/Amap API",
        "source_note": "這不是官方完整 TravelEval 分數；它輸出 paper-style metric table，去除高德 API 必要項目。",
        "score_direction_note": "↑ 表示越高越好，↓ 表示越低越好，- 表示成本分解值本身不代表好壞。",
        "computed_metrics": computed,
        "skipped_gaode_metrics": skipped,
        "currency": terms["budget_currency"],
        "cost_breakdown": cost_breakdown,
        "sandbox_summary": {
            "travel_eval_dir": str(TRAVELEVAL_DIR),
            "city_key": sandbox["city_key"],
            "matched_attractions": len(utility["records"]),
        },
    }


def render_partial_metrics_markdown(table: Dict[str, Any]) -> str:
    lines = [
        "# TravelEval Partial Metrics Without Gaode",
        "",
        table["source_note"],
        "",
        table["score_direction_note"],
        "",
        "## Computed Metrics",
        "",
        "| Dimension | Metric | Direction | Value | Status | Note |",
        "|---|---:|:---:|---:|---|---|",
    ]
    for row in table["computed_metrics"]:
        value = row["value"]
        if value is None:
            value_text = "NA"
        elif isinstance(value, float):
            value_text = f"{value:.4f}"
        else:
            value_text = str(value)
        note = compact(row.get("note"))
        lines.append(
            f"| {row['dimension']} | {row['metric']} | {row['direction']} | {value_text} | {row['status']} | {note} |"
        )

    lines.extend([
        "",
        "## Skipped Gaode Metrics",
        "",
        "| Dimension | Metric | Direction | Reason |",
        "|---|---:|:---:|---|",
    ])
    for row in table["skipped_gaode_metrics"]:
        lines.append(f"| {row['dimension']} | {row['metric']} | {row['direction']} | {compact(row.get('note'))} |")
    lines.append("")
    return "\n".join(lines)


def evaluate_without_gaode(final_result: Dict[str, Any], planner: Dict[str, Any], ai_plan: Dict[str, Any]) -> Dict[str, Any]:
    terms = planner_terms(planner)
    user_query = build_user_query(planner)
    sandbox = load_sandbox(terms["destination"])
    evidence = candidate_evidence(final_result)
    text = full_text(final_result)
    cost_breakdown = ai_plan.get("cost_breakdown") or {}
    total_cost = as_float(cost_breakdown.get("total"))
    budget = as_float(user_query.get("budget"))

    place_grounding = score_place_grounding(ai_plan, sandbox, evidence)
    opening_hours = score_opening_hours(ai_plan, sandbox)
    visit_duration = score_visit_duration(ai_plan, sandbox)
    schedule_validity = score_schedule_validity(ai_plan)
    daily_utilization = score_daily_utilization(ai_plan)
    pace = score_pace(ai_plan, terms["pace"])
    accommodation = score_accommodation(ai_plan, sandbox, terms["nights"])
    transport = score_transport_preference(ai_plan, terms["preferred_transportation"], final_result)
    key_area_coverage = term_coverage(terms["key_areas"], text)
    food_coverage = term_coverage(terms["food_preferences"], text)
    desired_coverage = desired_experience_coverage(terms["desired_experiences"], ai_plan, text)
    cost_completeness = score_cost_completeness(cost_breakdown)
    budget_efficiency = score_budget_efficiency(total_cost, budget)
    diversity = score_diversity(ai_plan)

    same_days = 1.0 if ai_plan["summary"].get("total_days") == user_query.get("days") else 0.0
    same_people = 1.0 if ai_plan["summary"].get("total_travelers") == user_query.get("people_number") else 0.0
    has_intercity_need = bool(terms["origin"] and terms["destination"] and normalize_text(terms["origin"]) != normalize_text(terms["destination"]))
    has_intercity_details = bool((ai_plan.get("intercity_transport") or {}).get("transport_type"))
    intercity_completeness = 1.0 if not has_intercity_need or has_intercity_details else 0.0

    budget_satisfaction = None
    if budget > 0 and total_cost > 0:
        budget_satisfaction = 1.0 if total_cost <= budget else 0.0

    price = (selected_hotel(final_result) or {}).get("price") or {}
    price_evidence = 1.0 if price.get("source") and price.get("availability_confirmed") else 0.5 if price else 0.0

    detailed_metrics = {
        "accuracy": {
            "cost_deviation_rate_score": metric(1.0, 0.0, "轉換後 cost_breakdown 與可計算成本一致。"),
            "people_deviation_score": metric(same_people, 1 - same_people),
            "place_grounding_score": metric(place_grounding["score"], place_grounding["value"], "使用 TravelEval 本地資料庫或候選 POI 座標驗證。", place_grounding["checked"]),
            "opening_hours_score": metric(opening_hours["score"], opening_hours["value"], "只檢查能在 TravelEval attractions CSV 匹配到的景點。", opening_hours["checked"]),
            "intercity_planning_completeness_proxy": metric(intercity_completeness, None, "不呼叫高德；只檢查跨城旅行是否有火車/航班明細。"),
        },
        "constraint": {
            "budget_satisfaction": metric(budget_satisfaction, {"budget": budget, "total_cost": total_cost}, "成本只包含目前 final result 可確認的價格。"),
            "time_compliance": metric(same_days, {"planned_days": ai_plan["summary"].get("total_days"), "requested_days": user_query.get("days")}),
            "people_adaptability": metric(same_people, {"planned_people": ai_plan["summary"].get("total_travelers"), "requested_people": user_query.get("people_number")}),
            "accommodation_satisfaction": metric(accommodation["score"], None, "檢查住宿夜數與本地住宿資料庫近似匹配。", accommodation),
            "transportation_satisfaction": metric(transport["score"], None, "不查路線；只看行程是否明確呈現偏好交通方式或至少保留通勤 slot。", transport),
            "diet_preference_satisfaction": metric(food_coverage["score"], None, "用最後行程文字檢查飲食偏好覆蓋。", food_coverage),
            "travel_time_satisfaction": metric(visit_duration["score"], None, "用 TravelEval 推薦停留時間做離線近似。", visit_duration["checked"]),
        },
        "time": {
            "schedule_order_validity": metric(schedule_validity["score"], None, "檢查時間格式、正時長與重疊。", schedule_validity["issues"]),
            "daily_time_utilization": metric(daily_utilization["score"], None, "以每日餐飲與景點活動時數 / 8 小時作為離線近似。", daily_utilization["days"]),
            "pace_alignment": metric(pace["score"], None, "用 5W1H pace 對每日景點數做近似。", pace["days"]),
        },
        "economy": {
            "budget_efficiency": metric(budget_efficiency["score"], budget_efficiency, "超預算時用 budget / known_total_cost 給部分分。"),
            "cost_breakdown_completeness": metric(cost_completeness["score"], None, "檢查住宿、餐飲、市內交通、城際交通、景點是否有價格。", cost_completeness),
            "lodging_price_evidence": metric(price_evidence, {"source": price.get("source"), "availability_confirmed": price.get("availability_confirmed")}),
        },
        "utility": {
            "key_area_coverage": metric(key_area_coverage["score"], None, "檢查 5W1H key_areas 是否出現在最後行程。", key_area_coverage),
            "food_preference_coverage": metric(food_coverage["score"], None, "小籠包允許湯包作為同義近似。", food_coverage),
            "desired_experience_coverage": metric(desired_coverage["score"], None, "檢查 desired_experiences 是否被最後行程文字覆蓋。", desired_coverage),
            "activity_diversity_proxy": metric(diversity["score"], None, "不用官方分類表時計算一個簡單活動多樣性近似。", diversity),
            "candidate_decision_grounding": metric(1.0 if final_result.get("used_candidate_ids") else 0.0, None, "最後結果有保留 used_candidate_ids。"),
        },
        "space": None,
    }

    dimension_scores = {}
    for dimension, metrics in detailed_metrics.items():
        if metrics is None:
            dimension_scores[dimension] = None
            continue
        dimension_scores[dimension] = average([item["score"] for item in metrics.values()])

    overall = average(list(dimension_scores.values()))
    skipped_metrics = [
        {
            "dimension": "space",
            "metric": "RP / CSM",
            "reason": "TravelEval 的 SpaceMetrics 會使用 GeoCalculator 與高德路線/距離能力，本次完全排除。",
        },
        {
            "dimension": "accuracy",
            "metric": "transportation_breaks",
            "reason": "TravelEval 會檢查相鄰 POI 的公共交通可達性，會碰到高德路線能力，本次排除。",
        },
    ]

    return {
        "status": "success",
        "score_scale": "0.0-1.0",
        "overall_score_without_gaode": overall,
        "overall_score_percent_without_gaode": round(overall * 100, 2) if overall is not None else None,
        "dimension_scores": dimension_scores,
        "detailed_metrics": detailed_metrics,
        "skipped_metrics": skipped_metrics,
        "converted_user_query": user_query,
        "sandbox_summary": {
            "travel_eval_dir": str(TRAVELEVAL_DIR),
            "city_key": sandbox["city_key"],
            "attractions_rows": len(sandbox.get("attractions") or []),
            "restaurants_rows": len(sandbox.get("restaurants") or []),
            "accommodations_rows": len(sandbox.get("accommodations") or []),
        },
        "security_note": "此離線評分器不讀取、不要求、不輸出高德 API key。",
    }


def run(final_path: Path, planner_path: Path, output_dir: Path, output_prefix: str) -> Dict[str, Any]:
    final_result = read_json(final_path)
    planner = read_json(planner_path)
    ai_plan = convert_to_traveleval_plan(final_result, planner)
    score_result = evaluate_without_gaode(final_result, planner, ai_plan)
    partial_metrics_table = build_partial_metrics_table(final_result, planner, ai_plan)

    plan_path = save_json(ai_plan, output_dir, f"{output_prefix}_traveleval_plan.json")
    partial_table_path = save_json(partial_metrics_table, output_dir, f"{output_prefix}_partial_metrics_without_gaode.json")
    partial_markdown_path = output_dir / f"{output_prefix}_partial_metrics_without_gaode.md"
    partial_markdown_path.write_text(render_partial_metrics_markdown(partial_metrics_table), encoding="utf-8")
    score_payload = {
        "source_files": {
            "final_result": str(final_path),
            "planner_result": str(planner_path),
            "converted_plan": str(plan_path),
            "partial_metrics_table": str(partial_table_path),
            "partial_metrics_markdown": str(partial_markdown_path),
        },
        "travel_eval_reference": {
            "evaluator": str(TRAVELEVAL_DIR / "core" / "evaluator.py"),
            "plan_extractor": str(TRAVELEVAL_DIR / "core" / "utils" / "plan_extractors.py"),
            "skipped_gaode_metrics": [
                str(TRAVELEVAL_DIR / "core" / "metrics" / "space.py"),
                str(TRAVELEVAL_DIR / "core" / "metrics" / "accuracy.py"),
            ],
        },
        "score_result": score_result,
        "partial_metrics_table": partial_metrics_table,
    }
    score_path = save_json(score_payload, output_dir, f"{output_prefix}_offline_score_without_gaode.json")
    return {
        "plan_path": str(plan_path),
        "score_path": str(score_path),
        "partial_table_path": str(partial_table_path),
        "partial_markdown_path": str(partial_markdown_path),
        "score": score_result,
        "partial_metrics_table": partial_metrics_table,
    }


if __name__ == "__main__":
    base_dir = PROJECT_DIR
    final_path = (
        Path(sys.argv[1]).resolve()
        if len(sys.argv) > 1
        else base_dir / "traced_outputs" / DEFAULT_FINAL_FILENAME
    )
    planner_path = (
        Path(sys.argv[2]).resolve()
        if len(sys.argv) > 2
        else base_dir / "traced_outputs" / DEFAULT_PLANNER_FILENAME
    )
    output_dir = EVALUATION_DIR / "outputs"
    output_prefix = sys.argv[3] if len(sys.argv) > 3 else (DEFAULT_OUTPUT_PREFIX or final_path.stem)

    result = run(final_path, planner_path, output_dir, output_prefix)

    print("=" * 20 + "TravelEval 離線評分（跳過高德）" + "=" * 20)
    print(json.dumps({
        "overall_score_without_gaode": result["score"]["overall_score_without_gaode"],
        "overall_score_percent_without_gaode": result["score"]["overall_score_percent_without_gaode"],
        "dimension_scores": result["score"]["dimension_scores"],
        "skipped_metrics": result["score"]["skipped_metrics"],
    }, ensure_ascii=False, indent=2))
    print("=" * 20 + "保存位置" + "=" * 20)
    print(result["plan_path"])
    print(result["score_path"])
    print(result["partial_table_path"])
    print(result["partial_markdown_path"])
