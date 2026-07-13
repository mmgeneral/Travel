from __future__ import annotations

import os
import json
import sys
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

# 2. 定義 MCP Tool
@mcp.tool()
def extract_searchable_activities(itinerary_json_str: str) -> str:
    """
    接收完整的旅遊行程 JSON 字串。
    透過 LLM 語意分析，篩選出「需要進一步查詢實際地點或店家」的模糊行程（如：特色小吃、夜市、商圈），
    並將這些行程打包成一個需要查詢的任務集合回傳。
    """
    
    # print("🔍 [MCP Tool 觸發] 正在分析行程中需要額外搜尋的項目...")
    
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

    請嚴格依照以下 JSON 格式輸出：
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
          "must_have": ["適合早午餐"]
        },
        {
          "slot_id": "day1_slot09",
          "day": 1,
          "day_of_week": "saturday",
          "start_time": "17:30",
          "end_time": "19:00",
          "time": "17:30~19:00",
          "slot_type": "meal",
          "original_activity": "品嚐台南夜市美食",
          "reason_for_search": "需確認當天哪個夜市有營業",
          "suggested_search_query": "台南 夜市 美食",
          "target_terms": ["夜市", "美食"],
          "location_terms": ["台南"],
          "intent_type": "night_market",
          "must_have": ["晚餐時段可安排"]
        }
      ]
    }
    """

    try:
        # 呼叫 LLM 進行語意分析與過濾
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"這是準備要分析的行程：\n{itinerary_json_str}"}
            ],
            response_format={ "type": "json_object" } # 強制輸出 JSON 格式
        )
        
        # 取得 LLM 分析後的 JSON 結果
        result = response.choices[0].message.content
        print("✅ [MCP Tool 完成] 成功提取待搜尋項目！")
        return result
        
    except Exception as e:
        return json.dumps({"error": f"分析行程時發生錯誤: {str(e)}"}, ensure_ascii=False)
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
