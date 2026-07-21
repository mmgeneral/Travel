from __future__ import annotations

import os
import json
import re
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()  # 載入 .env 檔案中的環境變數
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

SEARCH_MODEL = os.getenv("OPENAI_SEARCH_MODEL", os.getenv("OPENAI_PLANNER_MODEL", "gpt-4o"))
PLANNER_MODEL = os.getenv("OPENAI_PLANNER_MODEL", "gpt-4o")
WEB_SEARCH_CONTEXT_SIZE = os.getenv("OPENAI_WEB_SEARCH_CONTEXT_SIZE", "low")

WEEKDAY_ALIASES = {
    "monday": {"monday", "mon", "星期一", "禮拜一", "週一", "周一", "一"},
    "tuesday": {"tuesday", "tue", "tues", "星期二", "禮拜二", "週二", "周二", "二"},
    "wednesday": {"wednesday", "wed", "星期三", "禮拜三", "週三", "周三", "三"},
    "thursday": {"thursday", "thu", "thur", "thurs", "星期四", "禮拜四", "週四", "周四", "四"},
    "friday": {"friday", "fri", "星期五", "禮拜五", "週五", "周五", "五"},
    "saturday": {"saturday", "sat", "星期六", "禮拜六", "週六", "周六", "六"},
    "sunday": {"sunday", "sun", "星期日", "星期天", "禮拜日", "禮拜天", "週日", "周日", "週天", "周天", "日", "天"},
}


def _normalize_day_of_week(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text or text in {"none", "null", "unknown", "未知", "不確定"}:
        return None
    for canonical, aliases in WEEKDAY_ALIASES.items():
        if text == canonical or text in aliases:
            return canonical
    return None


def _normalize_iso2_code(value: object) -> str | None:
    text = str(value or "").strip().upper()
    if re.fullmatch(r"[A-Z]{2}", text):
        return text
    return None


def _time_parts(time_text: str) -> tuple[str | None, str | None]:
    parts = re.split(r"~|–|-|到|至", str(time_text or ""))
    if len(parts) < 2:
        return None, None
    return parts[0].strip() or None, parts[1].strip() or None


def _stable_slot_id(day: object, index: int) -> str:
    day_text = re.sub(r"\W+", "_", str(day or "unknown")).strip("_") or "unknown"
    return f"day{day_text}_slot{index + 1:02d}"


def _normalize_schedule_item(item: dict, day: object, index: int, day_of_week: str | None = None) -> dict:
    normalized = dict(item)
    normalized.setdefault("slot_id", _stable_slot_id(day, index))
    item_day_of_week = _normalize_day_of_week(normalized.get("day_of_week") or normalized.get("weekday"))
    normalized["day_of_week"] = item_day_of_week or day_of_week

    start_time = normalized.get("start_time")
    end_time = normalized.get("end_time")
    if (not start_time or not end_time) and normalized.get("time"):
        parsed_start, parsed_end = _time_parts(str(normalized.get("time")))
        start_time = start_time or parsed_start
        end_time = end_time or parsed_end

    if start_time:
        normalized["start_time"] = start_time
    if end_time:
        normalized["end_time"] = end_time
    if not normalized.get("time") and start_time and end_time:
        normalized["time"] = f"{start_time}~{end_time}"

    activity = str(normalized.get("activity") or "")
    slot_type = normalized.get("slot_type")
    if not slot_type:
        if normalized.get("is_commute") is True or "通勤" in activity:
            slot_type = "transport"
        elif any(term in activity for term in ["早餐", "午餐", "晚餐", "小吃", "美食"]):
            slot_type = "meal"
        else:
            slot_type = "activity"
        normalized["slot_type"] = slot_type

    normalized["is_commute"] = bool(
        normalized.get("is_commute") is True or slot_type == "transport" or "通勤" in activity
    )
    normalized.setdefault("is_searchable", False)
    normalized.setdefault("locked", False)
    normalized.setdefault("place_query", None)
    normalized.setdefault("route_entry_place", normalized.get("place_query"))
    normalized.setdefault("route_exit_place", normalized.get("place_query"))
    return normalized


def _normalize_itinerary_schema(result: dict) -> dict:
    normalized = dict(result)
    trip_context = normalized.get("trip_context")
    if not isinstance(trip_context, dict):
        trip_context = {}
    trip_context.setdefault("origin", "台灣")
    trip_context.setdefault("destination", None)
    country_code = _normalize_iso2_code(trip_context.get("country_code"))
    region_code = _normalize_iso2_code(trip_context.get("region_code"))
    trip_context["country_code"] = country_code
    trip_context["region_code"] = region_code or country_code
    trip_context.setdefault("language_code", "zh-TW")
    trip_context.setdefault("travel_start_date", None)
    trip_context.setdefault("arrival_anchor", None)
    trip_context.setdefault("return_anchor", None)
    trip_context.setdefault("lodging_anchor", None)
    normalized["trip_context"] = trip_context

    itinerary = normalized.get("itinerary", [])
    if isinstance(itinerary, list):
        for day_index, day_obj in enumerate(itinerary):
            if not isinstance(day_obj, dict):
                continue
            day = day_obj.get("day", day_index + 1)
            day_of_week = _normalize_day_of_week(day_obj.get("day_of_week") or day_obj.get("weekday"))
            day_obj["day_of_week"] = day_of_week
            schedule = day_obj.get("schedule", [])
            if not isinstance(schedule, list):
                continue
            day_obj["schedule"] = [
                _normalize_schedule_item(item, day, item_index, day_of_week=day_of_week)
                if isinstance(item, dict)
                else item
                for item_index, item in enumerate(schedule)
            ]
    return normalized


def generate_travel_itinerary_json(user_query: str) -> dict:
    """
    接收使用者查詢，透過 OpenAI 內建的原生 web_search 工具進行搜尋，
    並確保以合法的 JSON 格式回傳完整的旅遊行程。
    """
    
    system_prompt = """
    你是一位專業的旅遊規劃 Agent。請根據使用者的需求與內建 Web Search 的結果規劃行程。
    請務必以 JSON 格式輸出，且只能輸出合法的 JSON，不要包含任何額外的 Markdown 標記（如 ```json）。
    請嚴格遵守以下 JSON 結構與邏輯規則：

    1. **國內外判斷與航班規則**：
       - 預設出發地為台灣。請判斷 "destination_type" 為 "international" 或 "domestic"。
       - 若為 "international"（國外旅遊），第 1 天與最後 1 天可保留彈性航班 slot，但仍需符合下方 schedule schema。
       - 若為 "domestic"（國內旅遊），則正常安排所有天數的具體時間軸。

    2. **trip_context 結構**：
       - 必須輸出 trip_context，讓後續模組不用猜測起訖點。
       - trip_context 必須包含：
         {
           "origin": "台灣",
           "destination": "台南",
           "country_code": "TW",
           "region_code": "TW",
           "language_code": "zh-TW",
           "travel_start_date": null,
           "arrival_anchor": {"name": "台南火車站", "place_query": "台南火車站", "anchor_type": "arrival_station"},
           "return_anchor": {"name": "台南火車站", "place_query": "台南火車站", "anchor_type": "return_station"},
           "lodging_anchor": {"name": null, "place_query": null, "anchor_type": "lodging"}
         }
       - 若使用者沒有指定住宿，lodging_anchor 可為 null 欄位結構，但不可省略。
       - country_code / region_code 請使用目的地國家或地區的 ISO 3166-1 alpha-2 代碼；例如台灣 TW、日本 JP、韓國 KR。
       - language_code 請用使用者偏好的回覆語言；繁體中文使用 zh-TW。
       - 若使用者明確提供出發日期，travel_start_date 請用 YYYY-MM-DD；若無法判斷，請填 null。
       - arrival_anchor / return_anchor 應依目的地調整；例如京都使用京都車站，不要固定台南火車站。
    
       2.x. planning_brief_5w1h 結構：
        除了 trip_context 之外，必須輸出 planning_brief_5w1h，作為後續搜尋、候選評分、行程驗證與使用者修訂的共同約束摘要。
        planning_brief_5w1h 的用途不是取代 itinerary，而是說明這趟旅程的規劃依據、使用者偏好與成功條件。請根據使用者原始需求抽取資訊；若使用者沒有明確提供，請填 null 或空陣列，不要編造。
        planning_brief_5w1h 必須包含：

        {
        "who": {
            "traveler_count": null,
            "companions": [],
            "traveler_profile": null,
            "mobility_constraints": [],
            "special_needs": []
        },
        "where": {
            "origin": "台灣",
            "destination": "台南",
            "key_areas": [],
            "must_include_areas": [],
            "avoid_areas": []
        },
        "when": {
            "duration_days": null,
            "duration_nights": null,
            "travel_start_date": null,
            "travel_end_date": null,
            "season_or_month": null,
            "day_of_week_constraints": []
        },
        "what": {
            "must_visit": [],
            "desired_experiences": [],
            "activity_types": [],
            "food_preferences": [],
            "shopping_preferences": [],
            "lodging_preferences": []
        },
        "why": {
            "trip_goal": null,
            "success_criteria": [],
            "priority_order": []
        },
        "how": {
            "budget": null,
            "preferred_transportation": null,
            "pace": "moderate",
            "grounding_needs": [],
            "hard_constraints": [],
            "soft_preferences": []
        }
        }

        欄位填寫規則：

        - who.traveler_count：使用者有明確說人數才填數字；否則填 null。
        - who.companions：例如 family、partner、friends、children、elderly；沒有提到則空陣列。
        - who.traveler_profile：可填使用者明確自述，例如「第一次去東台灣」、「不熟悉當地」；沒有則 null。
        - where.key_areas：從使用者需求與目的地推得的主要區域，例如「太魯閣」、「花蓮市區」。
        - where.must_include_areas：使用者明確要求一定要去的區域。
        - where.avoid_areas：使用者明確說不要去的區域。
        - when.duration_days / duration_nights：若使用者說三天兩夜，請填 3 / 2。
        - when.travel_start_date：只有明確日期才填 YYYY-MM-DD；只有月份或季節時填 null，並把文字放到 season_or_month。
        - what.must_visit：使用者明確說想去或一定要去的景點，例如「太魯閣」。
        - what.desired_experiences：使用者想體驗的內容，例如「當地美食」、「東台灣自然景觀」、「歷史文化」。
        - what.activity_types：請用簡短英文類型，例如 nature、local_food、culture、shopping、museum、family、relaxation、night_market。
        - why.trip_goal：用一句話摘要這趟旅行的主要目的。
        - why.success_criteria：列出什麼樣的行程才算符合使用者期待。
        - why.priority_order：依重要性列出規劃優先順序，由低到高給出優先級，最高為1，最低為5
        - how.budget：使用者有明確預算才填數字或原文；否則 null。
        - how.preferred_transportation：使用者有指定交通方式才填；否則 null。
        - how.pace：只能是 relaxed / moderate / intensive 之一；若無法判斷，預設 moderate。
        - how.grounding_needs：列出後續必須查證或具體化的項目，例如 ["local_food", "opening_hours", "route_feasibility", "weather_sensitive_activity"]。
        - how.hard_constraints：使用者明確不可違反的限制。
        - how.soft_preferences：使用者偏好但可彈性調整的條件。

        重要規則：
        1. planning_brief_5w1h 必須和 trip_context、itinerary 一致。
        2. 不要因為 planning_brief_5w1h 而省略 itinerary。
        3. itinerary 仍然必須輸出每日 schedule、slot_id、start_time、end_time、slot_type、is_searchable、place_query、route_entry_place、route_exit_place、locked 等欄位。
        4. 如果 planning_brief_5w1h 中出現 must_visit 或 hard_constraints，itinerary 必須盡量反映；若無法安排，請在 reminders 中說明。
        5. 模糊活動若與 planning_brief_5w1h 的 desired_experiences 或 grounding_needs 有關，請將該 schedule item 的 is_searchable 設為 true。

    3. **行程時間軸格式**：
       - 每一個 schedule item 都必須有穩定 slot_id，格式如 "day1_slot01"。
       - 每一天的 itinerary day object 必須包含 day_of_week，值只能是 monday/tuesday/wednesday/thursday/friday/saturday/sunday 或 null。
       - 如果使用者沒有給日期、星期或足以推論星期幾的資訊，day_of_week 必須填 null，不要猜。
       - 若 day object 的 day_of_week 已知，該日所有 schedule item 也必須同步包含相同 day_of_week。
       - 每一個 schedule item 都必須拆分 start_time 與 end_time；time 可保留成 "HH:MM~HH:MM" 方便閱讀。
       - 景點與景點、活動與活動之間，必須插入一個獨立的通勤物件，slot_type 為 "transport"，is_commute 為 true。
       - 不要把多個真實地點塞在同一個 activity。若活動包含兩個地點，請拆成兩個 schedule item，中間插入通勤。
       - 每個 schedule item 必須包含：
         slot_id, day_of_week, start_time, end_time, time, activity, slot_type, is_commute, is_searchable, place_query, route_entry_place, route_exit_place, locked。
       - slot_type 請使用 activity / attraction / meal / shopping / transport / lodging / flight / flexible 之一。
       - 明確地點要填 place_query、route_entry_place、route_exit_place。例如赤崁樓的三個欄位都可填 "赤崁樓"。
       - 模糊活動才把 is_searchable 設為 true，place_query 可為 null。例如「台南特色小吃」「深度旅遊」「夜市美食」。
       - 通勤 slot 的 place_query、route_entry_place、route_exit_place 可為 null。

    4. **預期輸出的 JSON 結構範例**：
    {
      "destination_type": "domestic",
      "trip_context": {
        "origin": "台灣",
        "destination": "台南",
        "country_code": "TW",
        "region_code": "TW",
        "language_code": "zh-TW",
        "travel_start_date": null,
        "arrival_anchor": {"name": "台南火車站", "place_query": "台南火車站", "anchor_type": "arrival_station"},
        "return_anchor": {"name": "台南火車站", "place_query": "台南火車站", "anchor_type": "return_station"},
        "lodging_anchor": {"name": null, "place_query": null, "anchor_type": "lodging"}
      },
      "itinerary": [
        {
          "day": 1,
          "date_description": "第一天",
          "day_of_week": null,
          "schedule": [
            {
              "slot_id": "day1_slot01",
              "day_of_week": null,
              "start_time": "10:00",
              "end_time": "11:00",
              "time": "10:00~11:00",
              "activity": "享用台南特色牛肉湯作為早午餐",
              "slot_type": "meal",
              "is_commute": false,
              "is_searchable": true,
              "place_query": null,
              "route_entry_place": null,
              "route_exit_place": null,
              "locked": false
            },
            {
              "slot_id": "day1_slot02",
              "day_of_week": null,
              "start_time": "11:00",
              "end_time": "11:30",
              "time": "11:00~11:30",
              "activity": "預留通勤時間",
              "slot_type": "transport",
              "is_commute": true,
              "is_searchable": false,
              "place_query": null,
              "route_entry_place": null,
              "route_exit_place": null,
              "locked": false
            },
            {
              "slot_id": "day1_slot03",
              "day_of_week": null,
              "start_time": "11:30",
              "end_time": "13:00",
              "time": "11:30~13:00",
              "activity": "參觀赤崁樓",
              "slot_type": "attraction",
              "is_commute": false,
              "is_searchable": false,
              "place_query": "赤崁樓",
              "route_entry_place": "赤崁樓",
              "route_exit_place": "赤崁樓",
              "locked": false
            }
          ]
        }
      ],
      "reminders": [
        "天氣較炎熱，請準備防曬用品與水分補充。",
        "建議提前購買門票以節省排隊時間。"
      ]
    }
    """

    research_prompt = """
    你是一位旅遊資料研究員。請根據使用者需求使用 Web Search 蒐集規劃行程需要的背景資料。
    請用繁體中文輸出摘要，不要輸出 JSON。摘要應包含：
    1. 目的地類型：國內或國外。
    2. 使用者明確指定的景點或偏好。
    3. 可能適合排入行程的區域、餐飲、景點、活動。
    4. 需要注意的營業時間、公休日、交通或季節提醒。
    """

    research_response = client.responses.create(
        model=SEARCH_MODEL,
        input=[
            {"role": "system", "content": research_prompt},
            {"role": "user", "content": user_query},
        ],
        tools=[{"type": "web_search", "search_context_size": WEB_SEARCH_CONTEXT_SIZE}],
        tool_choice="required",
    )

    web_research_summary = research_response.output_text.strip()

    final_response = client.responses.create(
        model=PLANNER_MODEL,
        input=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"使用者需求：{user_query}\n\n"
                    "以下是 Web Search 後整理出的旅遊資料，請根據這些資料產生最終 JSON 行程：\n"
                    f"{web_research_summary}"
                ),
            },
        ],
        text={"format": {"type": "json_object"}},
    )

    final_text = final_response.output_text
    return _normalize_itinerary_schema(json.loads(final_text))
    # 2. 解析 Responses API 的陣列回傳結構
    # final_text = ""
    # # 取出 output 陣列的第一個項目 (通常是 message)，並比對 output_text
    # for block in response.output[0].content:
    #     if block.type == "output_text":
    #         final_text += block.text

    # return json.loads(final_text)

def save_planner_output(data: dict, output_dir: str, filename: str) -> str:
    """
    將 PlannerAgent 的輸出存成 JSON 檔。

    output_dir:
      - 可以是絕對路徑，例如 /Users/.../saved_outputs
      - 也可以是相對路徑，例如 saved_outputs/test_case
        相對路徑會以 PlannerAgent.py 所在資料夾為基準

    filename:
      - 例如 01_planner_agent_output.json
    """
    base_dir = Path(__file__).resolve().parent
    output_path = Path(output_dir).expanduser()

    if not output_path.is_absolute():
        output_path = base_dir / output_path

    output_path.mkdir(parents=True, exist_ok=True)

    file_path = output_path / filename
    file_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return str(file_path)


# ---------------------------------------------------------
# 4. 測試執行區塊
# ---------------------------------------------------------
if __name__ == "__main__":
    
    print("🚀 啟動行程生成 Agent 測試（使用 OpenAI 內建網頁搜尋）...\n")
    print("-" * 50)
    
    # 測試國內旅遊
    test_str = "北京上海測資"
    print(f"【測試任務：{test_str}】")
    test_query_1 = "我们两个人想从北京去上海玩3天，预算3400元人民幣"
    result_1 = generate_travel_itinerary_json(test_query_1)
    print(json.dumps(result_1, indent=2, ensure_ascii=False))
    
    print("\n" + "=" * 50 + "\n")

    save_planner_output(result_1, "traced_outputs", f"{test_str}_result.json")
    
    # 測試國外旅遊 (驗證第一天/最後一天規則)
    # print("【測試任務：安排日本東京三天兩夜行程】")
    # test_query_2 = "我要去日本東京玩三天兩夜，主要想去秋葉原跟淺草寺。"
    # result_2 = generate_travel_itinerary_json(test_query_2)
    # print(json.dumps(result_2, indent=2, ensure_ascii=False))
