# AnchorResolver.py Code Review 與逐行導讀

檔案：`corporate_with_gemini/AnchorResolver.py`

這支程式的任務是：對每個「模糊行程 slot」找出它前後最近的明確行程點，並用 Google Places 把這些前後站解析成座標。後面的 `CandidateScorer` 才能判斷候選點放進去後順不順路。

簡化流程：

```text
planner itinerary
  + pending_searches
  ↓
找出 pending slot 在 schedule 的位置
  ↓
往前找最近一個非通勤、非模糊 slot
往後找最近一個非通勤、非模糊 slot
  ↓
把前後站活動文字清理成 Google Places query
  ↓
查 Google Places，拿 place_id / address / lat / lng
  ↓
輸出 anchor_context
```

## Code Review：Hard-coding 與風險

### 1. 台南火車站寫成預設值，只適合目前台南 demo

位置：第 14-15、356-358、480-482 行。

```python
DEFAULT_START_ANCHOR = "台南火車站"
DEFAULT_END_ANCHOR = "台南火車站"
destination_hint: str = "台南"
```

這是目前使用者要求「第一天第一站 / 最後一天最後站預設台南火車站」時合理的快速版本。但如果使用者問「去京都三天」，預設還是台南火車站就會錯。

建議：

```text
短期：由 subMain 傳入 start_anchor/end_anchor/destination_hint。
中期：讓 PlannerAgent 輸出 origin_anchor / return_anchor。
長期：用 trip_context 結構保存出發地、抵達站、返程站、住宿點。
```

例如：

```json
{
  "destination": "京都",
  "start_anchor": "京都車站",
  "end_anchor": "京都車站"
}
```

### 2. 中文活動清理規則是 hard-coded heuristic

位置：第 17-48、119-149、152-174 行。

例如：

```python
COMMUTE_TERMS = {"預留通勤時間", "通勤時間", "交通時間"}
LEADING_PHRASES = ["參觀", "造訪", "探索", "夜遊", ...]
```

它會把：

```text
參觀赤崁樓 → 赤崁樓
探索安平古堡和安平樹屋 → 安平古堡 / 安平樹屋
```

這對目前中文 planner 有用，但缺點是：

- 換英文 itinerary 會失效。
- 換不同 LLM 用詞可能失效，例如「走訪」「拜訪」「體驗」未必都有列到。
- 有些詞可能是地名的一部分，硬切可能切錯。

建議：不要永遠靠字串清理。Planner 應該輸出結構化欄位：

```json
{
  "activity": "參觀赤崁樓",
  "slot_type": "attraction",
  "place_query": "赤崁樓",
  "is_commute": false
}
```

### 3. 複合活動用分隔符拆，順序是假設

位置：第 135-142、163-166 行。

目前：

```python
parts = re.split(r"和|與|及|、|/|／", text)
chosen = components[-1] if role == "previous" else components[0]
```

例子：

```text
探索安平古堡和安平樹屋
```

如果這個 slot 是模糊午餐的前一站，程式會取最後一個：

```text
previous anchor = 安平樹屋
```

如果這個 slot 是下一站，程式會取第一個：

```text
next anchor = 安平古堡
```

這很直覺，但仍然是 heuristic。因為 LLM 寫「德記洋行與台南美術館二館」不一定代表實際順序真的先德記洋行再美術館。

建議：之後 planner 要把複合活動拆成多個 schedule item，或提供：

```json
{
  "route_entry_place": "安平古堡",
  "route_exit_place": "安平樹屋"
}
```

### 4. 靠 day/time/activity 找 slot，缺少穩定 slot_id

位置：第 91-96、225-244、247-256 行。

現在用：

```text
day|time|activity
```

當 key。

如果 extractor 回傳的 activity 跟 planner 原文稍微不同，可能找不到 slot。雖然 `_matches_pending_item` 有 substring fallback，但也可能誤判。

建議：Planner 一開始就給每個 schedule item `slot_id`。

```json
{
  "slot_id": "day1_slot05",
  "time": "12:10~13:30",
  "activity": "在國華街享用午餐..."
}
```

Extractor 回傳同一個 `slot_id`，Resolver 就不用猜。

### 5. 只拿 Google Places 第一筆，可能選錯 anchor

位置：第 294-296 行。

```python
places = searcher.search_text(query, page_size=1)
place = _normalize_place(places[0], query, "anchor", 0) if places else None
```

例子：

```text
query = "台南 美術館"
```

Google 第一筆可能不是你原本 itinerary 想要的那個館。現在沒有檢查名稱相似度、行政區、類型，也沒有讓 LLM 或 scorer 判斷 anchor 是否可信。

建議：

```text
短期：page_size=3，選名稱最像 query 的。
中期：anchor 也做 confidence score。
長期：planner 直接保留 place_id，避免二次解析。
```

### 6. Import `_normalize_place` 是私有函式耦合

位置：第 9 行。

```python
from PlacesSearchAgent import GooglePlacesTextSearch, _normalize_place
```

底線開頭 `_normalize_place` 通常代表「這是模組內部用的 helper」。AnchorResolver 直接 import 它，代表兩支檔案耦合比較緊。

建議：把 normalize place 抽到共用檔案，例如：

```text
PlaceModels.py
normalize_place()
```

### 7. Cache key 還不夠完整

位置：第 95-96、282-315 行。

目前 cache key 是：

```text
destination_hint|query
```

例如：

```text
台南|台南 赤崁樓
```

還算可用，但沒有包含：

- language code
- region code
- API field mask
- Places API 版本

短期 OK，長期如果你改查詢設定，可能吃到舊 cache。

---

## 逐行導讀

### 第 1-7 行：匯入標準工具

```python
import argparse
from datetime import datetime
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
```

- 第 1 行：`argparse` 用來做 CLI。例：你可以在 terminal 執行 `python AnchorResolver.py itinerary.json pending.json --destination-hint 台南`。
- 第 2 行：`datetime` 用來記錄 anchor 解析時間，例如 `"resolved_at": "2026-07-06T02:52:26"`。
- 第 3 行：`json` 用來讀寫 JSON 檔。
- 第 4 行：`re` 是正規表示式，用來切字串，例如把 `"安平古堡和安平樹屋"` 用 `和` 切開。
- 第 5 行：`sys` 用來從 stdin 讀資料。
- 第 6 行：`Path` 讓路徑處理比較穩，不用手動串字串。
- 第 7 行：型別註記。`Optional[int]` 代表可能是 `int`，也可能是 `None`。

### 第 9 行：匯入 Google Places 搜尋與正規化 helper

```python
from PlacesSearchAgent import GooglePlacesTextSearch, _normalize_place
```

這行代表 AnchorResolver 會重用 `PlacesSearchAgent` 的搜尋器。

例子：

```python
searcher = GooglePlacesTextSearch(include_quality_fields=False)
places = searcher.search_text("台南 赤崁樓", page_size=1)
```

`_normalize_place` 會把 Google 原始格式整理成我們內部使用的格式：

```json
{
  "place_id": "...",
  "name": "赤崁樓",
  "latitude": 22.9974,
  "longitude": 120.2025
}
```

風險：`_normalize_place` 前面有底線，表示它原本不是設計給外部檔案使用。

### 第 12-15 行：預設儲存位置與台南預設 anchor

```python
STORE_DIR = Path(__file__).resolve().parent / "candidate_store"
DEFAULT_ANCHOR_CACHE_PATH = STORE_DIR / "anchor_cache.json"
DEFAULT_START_ANCHOR = "台南火車站"
DEFAULT_END_ANCHOR = "台南火車站"
```

- 第 12 行：找到目前檔案所在資料夾，再接 `candidate_store`。
- 第 13 行：預設 anchor cache 放在 `candidate_store/anchor_cache.json`。
- 第 14 行：第一天如果找不到前一站，預設台南火車站。
- 第 15 行：最後一天如果找不到下一站，預設台南火車站。

例子：如果第一個模糊 slot 是「10:00~11:00 台南早餐」，前面沒有景點，前站就用台南火車站。

這裡是 hard-coded，但符合你當時指定的台南需求。

### 第 17-27 行：判斷哪些活動太泛或是通勤

```python
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
```

`COMMUTE_TERMS` 用來判斷一個 schedule item 是不是通勤。Resolver 找 anchor 時會跳過通勤。

例子：

```json
{"time": "12:00~12:10", "activity": "預留通勤時間"}
```

這不是目的地，所以不該當 anchor。

`GENERIC_ACTIVITY_TERMS` 用來判斷活動是不是太抽象。

例子：

```text
午餐
美食
飯店
```

這些不是可直接查 Google Places 的明確地點。遇到它們時會 fallback 到預設 anchor。

### 第 28-48 行：活動文字前綴清理表

```python
LEADING_PHRASES = [
    "搭乘高鐵抵達",
    ...
    "在",
]
```

這個 list 的用途是把活動文字裡比較像動詞的前綴拿掉。

例子：

```text
參觀赤崁樓 → 赤崁樓
造訪林百貨 → 林百貨
夜遊花園夜市 → 花園夜市
在國華街享用午餐 → 國華街享用午餐
```

這是 hard-coded 中文規則。短期可用，長期應改成 planner 輸出 `place_query`。

### 第 51-58 行：讀 JSON 檔或 stdin

```python
def _read_json_from_path_or_stdin(path: Optional[str]) -> Dict[str, Any]:
    if path:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    raw = sys.stdin.read().strip()
    if not raw:
        raise SystemExit("Please pass a JSON path or pipe JSON through stdin.")
    return json.loads(raw)
```

- 第 51 行：定義 helper，輸入可以是檔案路徑，也可以沒有。
- 第 52-53 行：如果有路徑，就讀檔並 parse JSON。
- 第 55 行：沒有路徑，就從 stdin 讀。
- 第 56-57 行：stdin 也沒資料，就結束程式並提示。
- 第 58 行：把 stdin 字串 parse 成 JSON。

例子：

```bash
python AnchorResolver.py itinerary.json pending.json
```

會走第 52-53 行。

### 第 61-72 行：把不同格式統一成 JSON object

```python
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
    raise TypeError(...)
```

它接受三種輸入：

1. 已經是 dict：

```json
{"pending_searches": [...]}
```

2. list：

```json
[
  {"day": 1, "time": "12:10~13:30"}
]
```

會包成：

```json
{"pending_searches": [...]}
```

3. JSON 字串：

```python
'{"pending_searches": [...]}'
```

如果都不是，就丟 TypeError。

### 第 75-80 行：取出 pending_searches

```python
def _extract_pending_searches(raw: Any) -> List[Dict[str, Any]]:
    obj = _as_json_object(raw)
    pending = obj.get("pending_searches", obj)
    if not isinstance(pending, list):
        raise ValueError(...)
    return [item for item in pending if isinstance(item, dict)]
```

這段把 extractor 的結果整理成 list。

例子：

```json
{
  "pending_searches": [
    {"day": 1, "time": "12:10~13:30", "original_activity": "國華街午餐"}
  ]
}
```

最後回傳：

```python
[
  {"day": 1, "time": "12:10~13:30", "original_activity": "國華街午餐"}
]
```

### 第 83-96 行：文字正規化、slot key、cache key

```python
def _compact_spaces(text: Any) -> str:
    return " ".join(str(text or "").split())
```

把多餘空白壓成單一空白。

例子：

```text
"台南   赤崁樓" → "台南 赤崁樓"
```

```python
def _normalize_text(text: Any) -> str:
    return _compact_spaces(text).replace("臺", "台").lower()
```

把文字統一：

```text
臺南孔廟 → 台南孔廟
ABC → abc
```

```python
def _slot_key(day: Any, time: Any, activity: Any) -> str:
    return "|".join([str(day or ""), str(time or ""), _normalize_text(activity)])
```

產生 slot 的比對 key。

例子：

```text
1|12:10~13:30|在國華街享用午餐（牛肉湯、虱目魚肚等小吃）
```

```python
def _cache_key(query: str, destination_hint: str) -> str:
    return _normalize_text(f"{destination_hint}|{query}")
```

產生 Google Places 查詢 cache key。

例子：

```text
台南|台南 赤崁樓
```

### 第 99-111 行：讀寫 anchor cache

```python
def _load_cache(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}
```

如果 cache 檔不存在，回傳空 dict。存在就讀 JSON。如果 JSON 壞掉，也回傳空 dict，避免整個流程爆掉。

```python
def _save_cache(path: Path, cache: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
```

儲存 cache。`ensure_ascii=False` 讓中文不會變成 unicode escape。

### 第 114-149 行：判斷通勤、清括號、清前綴、拆複合活動、判斷抽象活動

```python
def _is_commute(activity: Any) -> bool:
    text = str(activity or "")
    return any(term in text for term in COMMUTE_TERMS)
```

如果活動文字含有「預留通勤時間」，就回傳 True。

```python
def _strip_parentheses(text: str) -> str:
    return re.sub(r"[（(].*?[）)]", "", text).strip()
```

移除括號內容。

例子：

```text
夜遊花園夜市（注意僅週四六日營業）→ 夜遊花園夜市
```

```python
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
```

反覆移除前綴。

例子：

```text
參觀台南孔廟 → 台南孔廟
返回飯店並準備返程 → 返程
```

這裡用 `while changed` 是因為可能有多個前綴連在一起。

```python
def _split_activity_components(text: str) -> List[str]:
    parts = re.split(r"和|與|及|、|/|／", text)
    cleaned = []
    for part in parts:
        item = _strip_leading_phrases(_strip_parentheses(part)).strip(" ，,。")
        if item:
            cleaned.append(item)
    return cleaned or [text]
```

把複合活動拆開。

例子：

```text
探索安平古堡和安平樹屋
→ ["安平古堡", "安平樹屋"]
```

```python
def _looks_generic_activity(text: str) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return True
    return normalized in GENERIC_ACTIVITY_TERMS
```

判斷是不是太抽象。

例子：

```text
"午餐" → True
"赤崁樓" → False
```

### 第 152-174 行：把活動轉成 Places query

```python
def _query_for_activity(...):
```

這是 AnchorResolver 的核心小工具之一。它把 schedule 裡的活動文字變成 Google Places query。

重要邏輯：

- 如果 `force_default=True`，直接用預設 anchor。
- 清掉動詞與括號。
- 如果是前一站，複合活動取最後一個。
- 如果是下一站，複合活動取第一個。
- 如果太抽象，就 fallback 到預設 anchor。
- 如果 query 裡沒有城市，就加上 `destination_hint`。

例子 1：

```python
_query_for_activity("參觀赤崁樓", "previous", "台南", "台南火車站")
→ "台南 赤崁樓"
```

例子 2：

```python
_query_for_activity("探索安平古堡和安平樹屋", "previous", "台南", "台南火車站")
→ "台南 安平樹屋"
```

例子 3：

```python
_query_for_activity("午餐", "previous", "台南", "台南火車站")
→ "台南火車站"
```

### 第 177-222 行：把 itinerary schedule 整理成 day → schedule map

```python
def _day_number(day_obj, fallback):
    return day_obj.get("day", fallback)
```

如果 day 物件有 `"day": 1`，就用 1；沒有就用 fallback。

```python
def _schedule_for_day(day_obj, fallback_day):
```

把一天的 schedule 每個 item 轉成內部格式：

```json
{
  "day": 1,
  "time": "10:30~12:00",
  "activity": "參觀赤崁樓",
  "schedule_index": 2
}
```

`schedule_index` 很重要，因為後面要往前/往後找。

```python
def _build_schedule_by_day(itinerary_json):
```

把整份 itinerary 變成：

```python
{
  1: [day1 schedule items],
  2: [day2 schedule items]
}
```

```python
def _get_day_schedule(schedule_by_day, day):
```

支援 `day` 是數字 1 或字串 `"1"`。避免 LLM 一次回 int，一次回 string 時找不到。

### 第 225-279 行：找 pending slot 位置與前後 anchor item

```python
def _pending_keys(pending_searches):
```

把所有模糊 slot 做成 key set。

用途：找前後 anchor 時，要跳過其他模糊 slot，因為模糊 slot 不是明確 anchor。

```python
def _matches_pending_item(schedule_item, pending):
```

判斷 schedule item 是否就是 extractor 找出的 pending item。

條件：

1. day 相同。
2. time 相同。
3. activity 完全相同，或彼此包含。

例子：

```text
schedule activity = 在國華街享用午餐（牛肉湯、虱目魚肚等小吃）
pending activity = 國華街享用午餐
```

因為有包含關係，所以可能判定為同一個。

```python
def _find_slot_position(schedule_by_day, pending):
```

在某一天 schedule 裡找到 pending slot 的 index。

例子：

```text
Day1 schedule index 4 = 國華街午餐
→ return (1, 4)
```

```python
def _is_searchable_slot(item, searchable_keys):
```

判斷某個 schedule item 是否也是模糊 slot。

```python
def _nearest_anchor_item(day_schedule, slot_index, direction, searchable_keys):
```

往前或往後找最近 anchor。

- `direction=-1`：往前找。
- `direction=1`：往後找。
- 跳過通勤。
- 跳過模糊 slot。

例子：

```text
赤崁樓
通勤
國華街午餐（模糊）
通勤
台南孔廟
```

對「國華街午餐」：

```text
previous anchor = 赤崁樓
next anchor = 台南孔廟
```

### 第 282-315 行：解析一個 anchor query 到 Google Place

```python
def _resolve_place(query, destination_hint, cache, searcher):
```

流程：

1. 用 query + destination_hint 做 cache key。
2. 如果 cache 有資料，直接回傳。
3. 沒 cache 就查 Google Places。
4. 只拿第一筆。
5. 成功才寫入 cache。

例子：

```python
_resolve_place("台南 赤崁樓", "台南", cache, searcher)
```

可能回傳：

```json
{
  "query": "台南 赤崁樓",
  "place": {
    "name": "赤崁樓",
    "latitude": 22.9974779,
    "longitude": 120.2025433
  },
  "cache_hit": false
}
```

如果下次再查同樣 query：

```json
{
  "cache_hit": true,
  ...
}
```

### 第 318-350 行：組出一個 anchor reference

```python
def _make_anchor_ref(...):
```

這個函式把「schedule item」變成完整 anchor 物件。

輸出包含：

```json
{
  "source": "schedule",
  "role": "previous",
  "day": 1,
  "time": "10:30~12:00",
  "activity": "參觀赤崁樓",
  "query": "台南 赤崁樓",
  "place": {...},
  "cache_hit": true,
  "warnings": []
}
```

如果 `item is None`，代表找不到前/後站，就用 default anchor。

例子：

```text
第一天第一個活動就是模糊早餐
→ previous anchor = 台南火車站
```

### 第 353-462 行：主函式 resolve_anchor_context

這是外部真正會呼叫的函式。

```python
pending_searches = _extract_pending_searches(pending_searches_json)
```

把 extractor 的結果整理成 list。

```python
schedule_by_day = _build_schedule_by_day(itinerary_json)
```

把 planner itinerary 整理成 day map。

```python
searchable_keys = _pending_keys(pending_searches)
```

把模糊 slot 做成 key set，方便跳過。

```python
cache = _load_cache(cache_path)
searcher = GooglePlacesTextSearch(include_quality_fields=False)
```

讀 anchor cache，並建立 Places searcher。`include_quality_fields=False` 是為了省費用，anchor 只需要座標，不需要評分/營業時間。

```python
day_order = list(schedule_by_day.keys())
first_day = day_order[0] if day_order else None
last_day = day_order[-1] if day_order else None
```

找第一天和最後一天，用來判斷是否要套用台南火車站邊界規則。

主迴圈：

```python
for pending in pending_searches:
```

對每一個模糊 slot 找前後 anchor。

如果找不到 slot：

```python
if day is None or slot_index is None:
```

就前後都用預設 anchor，並加入 warning。

如果找得到：

```python
previous_item = _nearest_anchor_item(..., -1, ...)
next_item = _nearest_anchor_item(..., 1, ...)
```

分別往前/往後找最近明確站點。

```python
previous_is_start_boundary = ...
next_is_end_boundary = ...
```

如果前站剛好是第一天第一個 schedule item，強制視為 start boundary。  
如果後站剛好是最後一天最後一個 schedule item，強制視為 end boundary。

這是為了符合你的規則：

```text
第一天第一個行程預設台南火車站
最後一天最後一個行程預設台南火車站
```

最後組出：

```json
{
  "slot_key": "...",
  "day": 1,
  "time": "12:10~13:30",
  "original_activity": "...",
  "previous_anchor": {...},
  "next_anchor": {...},
  "warnings": [...]
}
```

全部完成後：

```python
_save_cache(cache_path, cache)
```

把新查到的 anchor 寫入 cache。

回傳：

```json
{
  "anchor_context_version": "anchor_resolver_v0.1",
  "created_at": "...",
  "destination_hint": "台南",
  "default_start_anchor": "台南火車站",
  "default_end_anchor": "台南火車站",
  "distance_note": "...",
  "slot_anchors": [...]
}
```

### 第 465-473 行：把 anchor_context 存成 JSON 檔

```python
def save_anchor_context_output(anchor_context, output_dir, filename="03_anchor_context_output.json"):
```

這是輸出 helper。

例子：

```python
save_anchor_context_output(anchor_context, run_dir)
```

會建立：

```text
run_outputs/20260706_013645/03_anchor_context_output.json
```

### 第 476-505 行：CLI 入口

```python
def main() -> None:
```

讓這支檔案可以直接從 terminal 執行。

```python
parser.add_argument("itinerary", ...)
parser.add_argument("pending_searches", ...)
```

要求你傳入兩個檔案：

```bash
python AnchorResolver.py 01_planner_agent_output.json 02_extract_searchable_activities_output.json
```

```python
parser.add_argument("--destination-hint", default="台南")
parser.add_argument("--start-anchor", default=DEFAULT_START_ANCHOR)
parser.add_argument("--end-anchor", default=DEFAULT_END_ANCHOR)
parser.add_argument("--cache-path", default=str(DEFAULT_ANCHOR_CACHE_PATH))
parser.add_argument("--output-dir", ...)
```

這些是可選參數。

例子：

```bash
python AnchorResolver.py itinerary.json pending.json \
  --destination-hint 京都 \
  --start-anchor 京都車站 \
  --end-anchor 京都車站
```

第 487-496 行讀檔並呼叫主函式。

第 498-501 行決定輸出方式：

- 如果有 `--output-dir`，就存檔並印路徑。
- 如果沒有，就直接把完整 JSON 印到 terminal。

第 504-505 行：

```python
if __name__ == "__main__":
    main()
```

意思是：只有當你直接執行這個檔案時才跑 CLI；如果別的檔案 import 它，不會自動執行。

---

## 總結

這支程式目前是合理的 prototype，但確實有 hard-coding：

- 台南火車站與台南 destination 預設。
- 中文動詞/通勤/泛用活動詞清單。
- 複合活動用文字分隔推測 route endpoint。
- 用 day/time/activity 當 slot identity。
- Google Places anchor 只拿第一筆。

目前它適合用來把台南 demo 流程跑通。若要走向碩論或可泛化系統，下一步最重要的是讓 Planner 輸出更結構化的 schema：

```json
{
  "slot_id": "day1_slot05",
  "start_time": "12:10",
  "end_time": "13:30",
  "activity": "在國華街享用午餐（牛肉湯、虱目魚肚等小吃）",
  "slot_type": "meal",
  "is_commute": false,
  "is_searchable": true,
  "place_query": null,
  "route_entry_place": null,
  "route_exit_place": null,
  "locked": false
}
```

這樣 AnchorResolver 就能從「猜文字」進化成「讀結構」，hard-coding 會少很多。
