from __future__ import annotations

import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def extract_constraints_from_query(user_query: str) -> dict:
    """
    Parse a free‑text user query and return structured constraints.

    Returns
    -------
    dict
        {
            "raw_constraints": { ... },  # extracted fields (no nulls)
            "source_query": user_query
        }
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if not client.api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment / .env")

    system_prompt = (
        "你是一個旅遊行程規畫的約束抽取器。\n"
        "使用者用自然語言表達對行程的需求和限制。請從中抽取以下結構化欄位，"
        "如果沒有就省略（不要放 null）：\n"
        "- min_rating (0~5 的小數, 例如「至少 4.2 顆星」=> 4.2)\n"
        "- max_price_level (0~4 的整數, 對應 Google Places 的 price_level 0~4, "
        "使用者說「不要太貴」=> 2, 「便宜」=>1, 「中等」=>2, 「高級」=>3, 「高消費」=>4)\n"
        "- open_until (HH:MM 格式, 例如「營業到晚上 10 點」=> \"22:00\")\n"
        "- requires_air_conditioning (bool)\n"
        "- requires_parking (bool)\n\n"
        "只回傳 JSON 物件，不要 markdown 格式。\n"
        "範例：\n"
        "使用者：「找一間至少 4.3 星有冷氣、營業到晚上11點的燒烤店」\n"
        "回傳：{\"min_rating\": 4.3, \"requires_air_conditioning\": true, \"open_until\": \"23:00\"}\n\n"
        "使用者：「住宿便宜就好，要有停車位」\n"
        "回傳：{\"max_price_level\": 1, \"requires_parking\": true}\n\n"
        "使用者：「隨便都可以」\n"
        "回傳：{}"
    )

    user_content = f"原始使用者輸入：\n{user_query}"

    try:
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        result_str = response.choices[0].message.content or "{}"
        raw = json.loads(result_str)
    except Exception:
        raw = {}

    return {
        "raw_constraints": raw,
        "source_query": user_query,
    }
