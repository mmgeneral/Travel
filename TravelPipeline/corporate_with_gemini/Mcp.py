from __future__ import annotations

import os
import json
import sys
import re
from openai import OpenAI
try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    class FastMCP:
        def __init__(self, *_args, **_kwargs):
            pass

        def tool(self):
            def decorator(func):
                return func
            return decorator
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()  # 載入 .env 檔案中的環境變數
mcp = FastMCP("TravelAnalyzer")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def _missing_numeric_constraints(pending_searches: list) -> bool:
    """
    如果有任何 pending_search 的 must_have 內含有數字或價格/時間相關關鍵字
    但 structured_constraints 為空，回傳 True。
    """
    for ps in pending_searches:
        sc = ps.get("structured_constraints", {})
        if sc:
            continue
        must_have = ps.get("must_have", [])
        for h in must_have:
            if not isinstance(h, str):
                continue
            if re.search(r"\d+|元|點|以上|以下|營業到", h):
                return True
    return False


# 2. 定義 MCP Tool
@mcp.tool()
def extract_searchable_activities(
    itinerary_json_str: str,
    raw_constraints: dict | None = None,
) -> str:
    """
    接收完整的旅遊行程 JSON 字串。
    透過 LLM 語意分析，篩選出「需要進一步查詢實際地點或店家」的模糊行程
    （如：特色小吃、夜市、商圈），並將這些行程打包成一個需要查詢的任務集合回傳。

    可選參數 raw_constraints：由 extract_constraints_from_query 抽取的原始約束。
    """

    system_prompt = """
    你是一個精準的語意拆解 Agent。你的任務是分析使用者提供的「旅遊行程 JSON」，
    辨識出哪些行程是「明確的目的地（如：奇美博物館、赤崁樓）」，哪些是「需要進一步搜尋的概述（如：台南特色牛肉湯、夜市美食、秋葉原探索）」。

    請將所有「需要進一步搜尋的概述」挑選出來，並組成一個 JSON 陣列回傳。
    不要回傳明確的目的地，也不要包含任何 Markdown 格式（如 ```json）。
    若 schedule item 已經有 is_searchable=true，請優先把它視為需要搜尋的項目。
    回傳時必須保留原 schedule item 的 slot_id、day_of_week、start_time、end_time、time、slot_type，讓後續模組不用靠文字猜測位置。
    請同時輸出結構化搜尋線索，讓後續評分器不用靠硬寫的中文停用詞猜語意：
    - target_terms：這格真正想找的主體，例如 ["牛肉湯"]、["老屋咖啡"]、["歷史街區"]。
    - location_terms：地理限制或區域，例如 ["台南", "國華街"]、["名古屋", "榮"]。
    - intent_type：places 類型或意圖，例如 restaurant、cafe、museum、local_culture、shopping、night_market。
    - must_have：必要條件，例如 ["適合午餐", "可停留兩小時"]；如果沒有就空陣列。

    每個 pending_search 都必須包含 "structured_constraints" 欄位（可為空物件 {}）。
    該欄位可包含以下可量化欄位（不需要的就不填，不要放 null）：
      - min_rating (0~5 小數, 例 4.2)
      - max_price_level (0~4 整數)
      - open_until (HH:MM 格式)
      - requires_air_conditioning (bool)
      - requires_parking (bool)

    範例（第二筆展示有實際約束）：
    {
      "pending_searches": [
        {
          "slot_id": "day1_slot01",
          "day": 1,
          "day_of_week": "saturday",
          "start_time": "10:00",
          "end_time": "11:00",
          "time": "10:00~11:00",
          "slot_type": "meal",
          "original_activity": "享用台南特色牛肉湯作為早午餐",
          "reason_for_search": "缺乏具體店家",
          "suggested_search_query": "台南 必吃 牛肉湯 早午餐",
          "target_terms": ["牛肉湯"],
          "location_terms": ["台南"],
          "intent_type": "restaurant",
          "must_have": ["適合早午餐"],
          "structured_constraints": {}
        },
        {
          "slot_id": "day2_slot02",
          "day": 2,
          "day_of_week": "sunday",
          "start_time": "09:00",
          "end_time": "10:00",
          "time": "09:00~10:00",
          "slot_type": "meal",
          "original_activity": "飯店自助早餐",
          "reason_for_search": "",
          "suggested_search_query": "",
          "target_terms": [],
          "location_terms": [],
          "intent_type": "",
          "must_have": ["停車位", "中等價位"],
          "structured_constraints": {
            "requires_parking": true,
            "max_price_level": 2
          }
        }
      ]
    }
    """

    base_content = f"這是準備要分析的行程：\n{itinerary_json_str}"
    if raw_constraints and raw_constraints.get("raw_constraints"):
        raw_val = raw_constraints["raw_constraints"]
        base_content += f"\n\n【使用者原始約束參考基準】\n{json.dumps(raw_val, ensure_ascii=False, indent=2)}"
        base_content += (
            "\n如果行程裡某個 slot 明顯對應這些約束但你判斷有缺漏，優先採用這份參考基準的值。"
        )
    user_content = base_content

    def _call_llm(user_msg: str) -> str:
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content or "{}"
        except Exception as e:
            return json.dumps(
                {"error": f"分析行程時發生錯誤: {str(e)}"}, ensure_ascii=False
            )

    # 第一次呼叫
    result_str = _call_llm(user_content)
    try:
        data = json.loads(result_str)
    except json.JSONDecodeError:
        data = {"pending_searches": []}
    pending_searches = data.get("pending_searches", [])
    if not isinstance(pending_searches, list):
        pending_searches = []

    # 確保每個項目都有 structured_constraints 欄位
    for ps in pending_searches:
        if "structured_constraints" not in ps:
            ps["structured_constraints"] = {}

    # retry 邏輯
    if _missing_numeric_constraints(pending_searches):
        extra_reminder = (
            "先前 LLM 指出下列 slot 的 must_have 含有價格/時間相關文字，"
            "但 structured_constraints 為空，請務必根據 slot 語意補上適當的約束欄位。"
        )
        second_content = base_content + f"\n\n【提醒】{extra_reminder}"
        result_str = _call_llm(second_content)

    print("✅ [MCP Tool 完成] 成功提取待搜尋項目！")
    return result_str
def save_extractor_output(data: str | dict, output_dir: str, filename: str) -> str:
    """
    將 extract_searchable_activities 的輸出存成 JSON 檔。
    data 可以是 JSON 字串或 dict。
    """
    base_dir = Path(__file__).resolve().parent
    output_path = Path(output_dir).expanduser()

    if not output_path.is_absolute():
        output_path = base_dir / output_path

    output_path.mkdir(parents=True, exist_ok=True)

    if isinstance(data, str):
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            payload = {"raw_text": data}
    else:
        payload = data

    file_path = output_path / filename
    file_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return str(file_path)

def main(filename: str = None):
    if filename is None:
      if len(sys.argv) < 2:
          print("錯誤：請提供輸入檔案的名稱！")
          sys.exit(1)
      input_filename = sys.argv[1]
    else:
      input_filename = filename
    input_path = Path(input_filename)
    absolute_input_path = input_path.resolve() #確保轉成絕對路徑 後續再轉回來 統一作業
    append_text = "stage1"
    output_filename = f"{absolute_input_path.stem}_{append_text}.json"

    try:
        with open(absolute_input_path, 'r', encoding='utf-8') as f:
            json_str_content = f.read()
    except FileNotFoundError:
        print(f"錯誤：找不到檔案 {absolute_input_path}")
        sys.exit(1)

    processed_result = extract_searchable_activities(json_str_content)
    try:
        processed_payload = json.loads(processed_result)
    except json.JSONDecodeError:
        processed_payload = {"raw_text": processed_result}

    # with open(output_filepath, 'w', encoding='utf-8') as f_out:
    #     json.dump(processed_payload, f_out, ensure_ascii=False, indent=4)
    saved_path = save_extractor_output(
        processed_payload,
        output_dir=str(absolute_input_path.parent),
        filename=output_filename
    )

    print(f"✅ 處理完成！結果已儲存至：\n{saved_path}")

# 3. 啟動 MCP 伺服器
if __name__ == "__main__":
    # MCP 伺服器預設透過 stdio (標準輸入輸出) 與 Client 端溝通
    # 如果你是要用 Claude Desktop 或自己寫的 Agent 掛載這個 MCP，就直接執行這行
    # mcp.run()
    main("corporate_with_gemini/traced_outputs/北京上海測資_result.json")
