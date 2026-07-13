from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv(override=True)

FINAL_ITINERARY_MODEL = os.getenv("FINAL_ITINERARY_MODEL", "gpt-4o")


def generate_final_itinerary(candidate_llm_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Let the final LLM decide lodging and fuzzy-slot replacements.
    Input should be CandidateScorer's *_candidate_llm_input.json.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in environment or .env")

    system_prompt = """
你是 Final Itinerary Agent。你的任務是根據 CandidateScorer 提供的候選景點與住宿，產生最終定案行程。

你必須遵守：
1. 只能使用輸入中的 candidate_slots.top_candidates 與 lodging_candidates，不要發明新景點或新飯店。
2. 每個模糊 slot 都要做決策：single_replace / minor_adjustment / major_rebuild。
3. 住宿要做決策：single_hotel / split_hotels。若 lodging_strategy 建議 single_hotel，除非理由非常強，否則不要換飯店。
4. final_itinerary 必須保留原本 itinerary 的 day / schedule 結構與時間軸；若只是替換模糊詞，盡量保留原 slot_id。
5. 替換模糊 slot 時，把 is_searchable 設為 false，並補上 place_query、route_entry_place、route_exit_place。
6. 若需要大改，只能根據已提供候選重排，不要新增未提供的店家或景點。
7. final_itinerary 裡每一個 schedule item 都要有 cost_estimate。幣別優先用使用者預算幣別；若無法判斷，使用 CNY。
8. cost_estimate 是粗估即可，但必須能用於預算加總。免費景點請給 party_total: 0，不要省略。
9. 請輸出合法 JSON，不要 Markdown。
10. selected_lodging_by_night.hotel.price 必須直接複製 lodging_candidates 中該飯店的 price object，不要自行換算、改寫或猜測住宿價格。

cost_estimate 格式：
{
  "currency": "CNY",
  "per_person": 0,
  "party_total": 0,
  "source": "llm_estimated | candidate_price | serpapi_google_hotels | free",
  "confidence": 0.0,
  "note": "估價依據"
}

估價原則：
- attraction：估門票；免費景點 party_total = 0。
- meal：估整團餐費。
- transport：估整團市內交通費。
- lodging：若是入住/住宿 slot，使用 selected_lodging_by_night 的住宿價格；若只是返回酒店且不新增費用，party_total = 0。
- intercity_transport：若 final_itinerary 有城際交通 slot，估整團高鐵/機票費。

輸出格式：
{
  "final_itinerary_version": "final_itinerary_agent_v0.1",
  "created_at": "ISO datetime",
  "decision_summary": {
    "overall_strategy": "single_replace | minor_adjustment | major_rebuild",
    "lodging_decision": "single_hotel | split_hotels",
    "reasoning": [],
    "warnings": []
  },
  "selected_lodging_by_night": [
    {
      "night": 1,
      "day_after": 1,
      "hotel": {
        "place_id": "...",
        "property_token": "...",
        "name": "...",
        "latitude": 0,
        "longitude": 0,
        "price": {}
      },
      "reason": "..."
    }
  ],
  "slot_decisions": [
    {
      "slot_id": "...",
      "day": 1,
      "original_activity": "...",
      "decision": "single_replace | minor_adjustment | major_rebuild",
      "selected_candidate": {
        "place_id": "...",
        "name": "...",
        "address": "...",
        "latitude": 0,
        "longitude": 0
      },
      "reason": "..."
    }
  ],
  "final_itinerary": [
    {
      "day": 1,
      "date_description": "...",
      "schedule": [
        {
          "slot_id": "...",
          "start_time": "09:00",
          "end_time": "10:00",
          "time": "09:00~10:00",
          "activity": "...",
          "slot_type": "attraction | meal | transport | lodging | other",
          "is_commute": false,
          "is_searchable": false,
          "place_query": "...",
          "route_entry_place": "...",
          "route_exit_place": "...",
          "cost_estimate": {
            "currency": "CNY",
            "per_person": 0,
            "party_total": 0,
            "source": "llm_estimated",
            "confidence": 0.6,
            "note": "..."
          }
        }
      ]
    }
  ],
  "reminders": [],
  "used_candidate_ids": {
    "places": [],
    "lodging": []
  }
}
"""

    user_payload = {
        "candidate_llm_input": candidate_llm_input,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=FINAL_ITINERARY_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content or "{}"
    result = json.loads(content)
    result.setdefault("final_itinerary_version", "final_itinerary_agent_v0.1")
    result.setdefault("created_at", datetime.now().isoformat(timespec="seconds"))
    repair_selected_lodging_prices(result, candidate_llm_input)
    return result


def _compact(value: Any) -> str:
    return " ".join(str(value or "").split()).lower()


def _lodging_keys(hotel: Dict[str, Any]) -> list[str]:
    keys = []
    for field in ("property_token", "place_id", "name"):
        value = _compact(hotel.get(field))
        if value:
            keys.append(value)
    return keys


def repair_selected_lodging_prices(result: Dict[str, Any], candidate_llm_input: Dict[str, Any]) -> None:
    lodging_index = {}
    for candidate in candidate_llm_input.get("lodging_candidates") or []:
        for key in _lodging_keys(candidate):
            lodging_index[key] = candidate

    for stay in result.get("selected_lodging_by_night") or []:
        hotel = stay.get("hotel") or {}
        candidate = None
        for key in _lodging_keys(hotel):
            candidate = lodging_index.get(key)
            if candidate:
                break
        if not candidate:
            continue

        if isinstance(candidate.get("price"), dict):
            hotel["price"] = dict(candidate["price"])
        for field in ("place_id", "property_token", "name", "address", "latitude", "longitude", "rating", "user_rating_count", "hotel_class"):
            if candidate.get(field) is not None:
                hotel[field] = candidate[field]


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
        else base_dir / "traced_outputs" / "北京上海測資_result_candidate_llm_input.json"
    )

    with open(input_path, "r", encoding="utf-8") as f:
        candidate_context = json.load(f)

    print("=" * 20 + "看一下輸入檔" + "=" * 20)
    print(input_path)
    print("=" * 20 + "結束" + "=" * 20)
    print("\n")

    final_result = generate_final_itinerary(candidate_context)

    print("=" * 20 + "看一下決策摘要" + "=" * 20)
    print(json.dumps(final_result.get("decision_summary", {}), ensure_ascii=False, indent=2))
    print("=" * 20 + "結束" + "=" * 20)
    print("\n")

    filename = f"{input_path.stem}_final_itinerary.json"
    output_path = base_dir / "traced_outputs"
    saved_path = save_json_output(final_result, output_path, filename, "final_itinerary")

    print("=" * 20 + "看一下保存位置" + "=" * 20)
    print(saved_path)
    print("=" * 20 + "結束" + "=" * 20)
