# PlacesSearchAgent.py Code Review 與逐行導讀

檔案：`corporate_with_gemini/PlacesSearchAgent.py`

這支程式的任務是：把 Ambiguity Extractor 找出的模糊行程 slot，轉成多組 Google Places Text Search 查詢，呼叫 Google Places API 找候選地點，最後整理成候選池給 `CandidateScorer` 評分。

更新註記：後續已依 review 修正三點：

- 語言與地區改為優先讀 `trip_context.language_code` / `trip_context.region_code`。
- Query generator prompt 已改成中性例子，不再使用台南牛肉湯示範。
- `search_places_for_pending_searches` 已可接收 `anchor_context`，並用前後 anchor 中心點建立 Google Places `locationBias`。

簡化流程：

```text
pending_searches
  ↓
每個模糊 slot 產生 3-4 組 Places query
  ↓
每組 query 呼叫 Google Places Text Search
  ↓
把 Google 原始 place 格式 normalize 成內部格式
  ↓
同一地點去重並合併 matched_queries
  ↓
輸出 candidate_slots
```

例子：

```json
{
  "slot_id": "day1_slot05",
  "original_activity": "在國華街享用午餐（牛肉湯、虱目魚肚等小吃）",
  "suggested_search_query": "國華街 牛肉湯 虱目魚肚 午餐 推薦"
}
```

會被轉成：

```json
[
  {"query": "台南 國華街 牛肉湯", "intent": "restaurant"},
  {"query": "台南 國華街 虱目魚肚", "intent": "restaurant"},
  {"query": "台南 中西區 小吃", "intent": "restaurant"},
  {"query": "台南 國華街 午餐", "intent": "restaurant"}
]
```

再去 Google Places 找候選店家。

---

## Code Review：Hard-coding 與風險

### 1. Google Places endpoint 寫死

位置：第 17 行。

```python
TEXT_SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"
```

這代表目前這支程式只支援 Google Places API New 的 Text Search。短期合理，因為我們現在要接 Places API。但長期如果想替換成 Geoapify / OSM / Overpass / SerpApi，就會卡住。

建議：

```text
短期：保留。
中期：抽成 provider interface，例如 PlaceSearchProvider。
長期：讓 Candidate Search 可切 GooglePlacesProvider / OSMProvider / GeoapifyProvider。
```

### 2. 語言與地區固定台灣繁中

位置：第 18-19、209-210 行。

```python
DEFAULT_LANGUAGE_CODE = "zh-TW"
DEFAULT_REGION_CODE = "TW"
```

這對台灣/台南 demo 很合理，但如果使用者要去日本、韓國、歐洲，地區 bias 仍是 `TW` 可能會影響搜尋結果。

例子：

```text
query = "京都 拉麵"
region_code = "TW"
```

Google 仍可能找得到京都，但搜尋語境不是最自然。

建議：從 `trip_context` 推出 `region_code` / `language_code`。

```json
{
  "destination": "京都",
  "country_code": "JP",
  "language_code": "zh-TW"
}
```

### 3. Query generator prompt 仍有台南例子

位置：第 108-126 行。

```text
像「台南 中西區 牛肉湯」或「台南 老屋咖啡」
```

這只是範例，不一定會讓模型只能產生台南，但 prompt 會有示範偏差。若目的地是京都，模型仍可能表現正常，因為 user_payload 有 `destination_hint`，但更乾淨的做法是把例子寫成中性。

建議：

```text
像「目的地 區域 食物類型」或「目的地 景點類型」
```

### 4. fallback query 還是偏簡單

位置：第 73-92 行。

目前 fallback 只用：

```text
suggested_search_query
original_activity
destination_hint + original_activity
destination_hint + suggested_search_query
```

這在沒有 OpenAI API 或 query generation 失敗時可保底，但不夠聰明。它不會根據 `slot_type`、`trip_context`、時間、anchor 附近區域產生更精準 query。

例子：

```text
original_activity = "深度旅遊"
destination_hint = "台南"
```

fallback 可能只得到：

```text
台南 深度旅遊
```

這很可能太泛。

建議：fallback 也讀 `slot_type` 與前後 anchor 的行政區/地名。

### 5. max query / max result 是固定預設

位置：第 98、333-334 行。

```python
max_queries: int = 4
max_queries_per_slot: int = 4
max_results_per_query: int = 6
```

也就是一個模糊 slot 最多：

```text
4 queries * 6 results = 24 raw candidates
```

這是成本與品質的折衷，但不是所有 slot 都適合。

例子：

```text
「安平豆花」可能 2 queries 就夠。
「台南深度旅遊」可能 6 queries 才夠。
```

建議：依 slot 模糊程度調整 query 數。越模糊，候選池可稍大；越明確，候選池可小。

### 6. Field mask 固定，且 quality fields 可能影響費用

位置：第 22-41、199-203 行。

`BASE_FIELD_MASK` 是基本地點資料。`QUALITY_FIELD_MASK` 額外拿評分、評論數、營業時間、商家狀態、網站。

這些欄位對 scoring 很有用，但外部 API 的欄位選擇會影響成本與 SKU，正式使用時要跟官方文件確認。

建議：

```text
開發/replay：include_quality_fields=false
正式 scoring：需要 quality/opening 時再打 include_quality_fields=true
長期：把 field profile 做成配置，例如 ids_only / basic / scoring。
```

### 7. 沒有 location bias

位置：第 211、226-227 行。

程式支援 `location_bias`，但目前 `search_places_for_pending_searches` 沒傳。這代表搜尋只靠文字，不會利用前後 anchor 附近座標。

例子：

```text
query = "咖啡"
anchor = 赤崁樓 / 台南孔廟
```

如果有 location bias，可以讓 Google 更偏向附近咖啡店。

建議：等 AnchorResolver 出來後，把 slot anchor context 傳進 Candidate Search，用前後 anchor 的中心點做 location bias。

### 8. errors 被收集但不會中斷

位置：第 352-362 行。

這是好事，因為單一 query 失敗不會讓整個流程爆掉。但目前沒有錯誤分類、重試、429/backoff。

建議：

```text
遇到 429：應該退避重試或暫停。
遇到 403：應該明確提示 API key / API restriction。
遇到 DNS：應該提示網路環境。
```

### 9. main() 還有台南 fallback

位置：第 400-403 行。

```python
destination_hint=os.getenv("TRAVEL_DESTINATION_HINT", "台南")
```

這只影響直接執行 `PlacesSearchAgent.py` 的測試模式，不影響 `subMain.py` 的新長期流程。但仍是 hard-coded 台南 fallback。

建議：改成空字串或從輸入 payload 的 `trip_context.destination` 取得。

---

## 逐行導讀

### 第 1-4 行：匯入標準工具

```python
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Optional
```

- 第 1 行：`json` 用來處理 JSON 字串與 dict 轉換。
- 第 2 行：`os` 用來讀環境變數，例如 `OPENAI_API_KEY`、`GOOGLE_PLACES_API_KEY`。
- 第 3 行：`sys` 用來從 stdin 讀資料。
- 第 4 行：typing 註記，讓函式輸入輸出更清楚。

例子：

```python
os.getenv("GOOGLE_PLACES_API_KEY")
```

會從 `.env` 或環境變數讀 Google Places key。

### 第 6-7 行：匯入外部套件

```python
import requests
from dotenv import load_dotenv
```

- 第 6 行：`requests` 用來發 HTTP POST 給 Google Places。
- 第 7 行：`load_dotenv` 讓程式讀 `.env` 裡的 API key。

例子：

```python
requests.post(TEXT_SEARCH_URL, headers=..., json=body)
```

會真的呼叫 Google Places API。

### 第 9-12 行：嘗試匯入 OpenAI

```python
try:
    from openai import OpenAI
except Exception:
    OpenAI = None
```

這裡讓程式在沒有 OpenAI 套件時仍能跑 fallback query。

例子：

```text
如果本機沒裝 openai package
→ OpenAI = None
→ generate_places_queries 會走 _fallback_queries
```

這是彈性設計，不算 hard-coding。

### 第 15 行：載入 .env

```python
load_dotenv()
```

讓後面可以讀：

```text
OPENAI_API_KEY
GOOGLE_PLACES_API_KEY
GOOGLE_MAPS_API_KEY
GOOGLE_PLACES_INCLUDE_QUALITY_FIELDS
```

### 第 17-20 行：全域預設設定

```python
TEXT_SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"
DEFAULT_LANGUAGE_CODE = "zh-TW"
DEFAULT_REGION_CODE = "TW"
DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
```

- 第 17 行：Google Places Text Search API endpoint。
- 第 18 行：回傳語言預設繁中。
- 第 19 行：搜尋地區預設台灣。
- 第 20 行：query generator 預設用 `OPENAI_MODEL`，如果沒有設定就用 `gpt-4o-mini`。

例子：

```text
query = "台南 牛肉湯"
language_code = "zh-TW"
region_code = "TW"
```

Google 回傳比較可能是繁中地址與台灣語境。

hard-coding：`zh-TW` / `TW` 對目前台南 demo 合理，但不適合所有國家。

### 第 22-31 行：基本 field mask

```python
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
```

Google Places API New 要求用 `X-Goog-FieldMask` 指定想拿哪些欄位。

這裡的基本欄位用途：

- `places.id`：Google place id。
- `places.name`：resource name，像 `places/ChIJ...`。
- `places.displayName`：使用者看得懂的名稱，例如「國華牛肉湯」。
- `places.formattedAddress`：地址。
- `places.location`：經緯度。
- `places.primaryType`：主要類型。
- `places.types`：所有類型。
- `places.googleMapsUri`：Google Maps 連結。

例子 normalize 後會變：

```json
{
  "place_id": "ChIJ...",
  "name": "國華牛肉湯",
  "address": "70055臺南市中西區府前里國華街二段140號",
  "latitude": 22.991037,
  "longitude": 120.1969604
}
```

### 第 33-41 行：評分用 field mask

```python
QUALITY_FIELD_MASK = [
    "places.businessStatus",
    "places.currentOpeningHours",
    "places.regularOpeningHours",
    "places.rating",
    "places.userRatingCount",
    "places.websiteUri",
]
```

這些欄位主要給 `CandidateScorer` 使用。

- `businessStatus`：店是否正常營業。
- `currentOpeningHours`：當前營業時間。
- `regularOpeningHours`：一般營業時間。
- `rating`：Google 評分。
- `userRatingCount`：評論數。
- `websiteUri`：官方網站。

例子：

```json
{
  "rating": 4.6,
  "user_rating_count": 534,
  "regular_opening_hours": {...}
}
```

注意：這些欄位可能影響 API 成本。正式實驗要記錄 field mask profile。

### 第 44-48 行：讀取布林環境變數

```python
def _truthy_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}
```

這個 helper 把 `.env` 裡的字串轉成 boolean。

例子：

```env
GOOGLE_PLACES_INCLUDE_QUALITY_FIELDS=true
```

會變成：

```python
True
```

如果沒有設定，就回傳 `default`。

### 第 51-62 行：把輸入轉成 JSON object

```python
def _as_json_object(raw: Any) -> Dict[str, Any]:
```

這跟 AnchorResolver 類似，支援三種輸入：

1. dict：

```json
{"pending_searches": [...]}
```

2. list：

```json
[{"original_activity": "台南特色小吃"}]
```

會包成：

```json
{"pending_searches": [...]}
```

3. JSON 字串：

```python
'{"pending_searches": [...]}'
```

如果不是這些格式，就丟 TypeError。

### 第 65-70 行：抽出 pending_searches

```python
def _extract_pending_searches(raw: Any) -> List[Dict[str, Any]]:
    obj = _as_json_object(raw)
    pending = obj.get("pending_searches", obj)
    if not isinstance(pending, list):
        raise ValueError(...)
    return [item for item in pending if isinstance(item, dict)]
```

這會把 extractor 結果統一成 list。

例子輸入：

```json
{
  "pending_searches": [
    {
      "slot_id": "day1_slot05",
      "original_activity": "國華街午餐"
    }
  ]
}
```

輸出：

```python
[
  {"slot_id": "day1_slot05", "original_activity": "國華街午餐"}
]
```

### 第 73-92 行：fallback query 產生器

```python
def _fallback_queries(task: Dict[str, Any], destination_hint: str = "") -> List[Dict[str, str]]:
```

這是在 OpenAI query generation 失敗或不可用時的備援。

第 74-76 行：

```python
original = str(task.get("original_activity") or "").strip()
suggested = str(task.get("suggested_search_query") or "").strip()
reason = str(task.get("reason_for_search") or "").strip()
```

從 pending slot 拿：

- 原始活動文字。
- extractor 建議查詢。
- 為什麼要查。

例子：

```json
{
  "original_activity": "在國華街享用午餐（牛肉湯、虱目魚肚等小吃）",
  "suggested_search_query": "國華街 牛肉湯 虱目魚肚 午餐 推薦",
  "reason_for_search": "缺乏具體店家資訊"
}
```

第 77 行：

```python
base_items = [suggested, original]
```

先把 suggested query 和原活動放進候選查詢。

第 79-82 行：

```python
if destination_hint and original and destination_hint not in original:
    base_items.append(f"{destination_hint} {original}")
if destination_hint and suggested and destination_hint not in suggested:
    base_items.append(f"{destination_hint} {suggested}")
```

如果原本字串沒有目的地，就補上。

例子：

```text
destination_hint = "台南"
original = "國華街午餐"
→ "台南 國華街午餐"
```

第 84-92 行：去除空字串與重複，最多回傳 4 筆。

```python
queries.append({"query": normalized, "intent": reason or "模糊行程候選搜尋"})
return queries[:4]
```

風險：這個 fallback 很簡單，沒有根據 slot_type 或前後 anchor 生成更聰明 query。

### 第 95-177 行：用 LLM 產生 Places queries

```python
def generate_places_queries(...):
```

這是把一個模糊 slot 轉成多個 Google Places 查詢的核心函式。

第 95-100 行參數：

```python
task
destination_hint = ""
max_queries = 4
model = DEFAULT_OPENAI_MODEL
```

例子：

```python
generate_places_queries(task, destination_hint="台南", max_queries=4)
```

第 105-106 行：

```python
if OpenAI is None or not os.getenv("OPENAI_API_KEY"):
    return _fallback_queries(...)
```

如果沒有 OpenAI，就走 fallback。

第 108-126 行：query generator prompt。

它要求 LLM：

- 把模糊活動轉成 3-4 組短查詢。
- 優先包含城市或區域。
- 不要產生網頁搜尋 query，例如 PTT、排行、推薦文章。
- 回傳 JSON。

例子輸出：

```json
{
  "queries": [
    {"query": "台南 國華街 牛肉湯", "intent": "restaurant"},
    {"query": "台南 國華街 虱目魚肚", "intent": "restaurant"}
  ]
}
```

第 128-142 行：組 user_payload。

```python
user_payload = {
    "destination_hint": destination_hint,
    "max_queries": max_queries,
    "ambiguous_slot": {...}
}
```

這裡現在有保留：

- `slot_id`
- `day`
- `start_time`
- `end_time`
- `time`
- `slot_type`
- `original_activity`
- `reason_for_search`
- `suggested_search_query`

例子：

```json
{
  "destination_hint": "台南",
  "max_queries": 4,
  "ambiguous_slot": {
    "slot_id": "day1_slot05",
    "slot_type": "meal",
    "original_activity": "國華街午餐"
  }
}
```

第 144-153 行：呼叫 OpenAI。

```python
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
response = client.chat.completions.create(...)
```

使用 JSON mode：

```python
response_format={"type": "json_object"}
```

這表示模型必須回 JSON object。

第 154-156 行：取出模型輸出並 parse。

```python
content = response.choices[0].message.content or "{}"
parsed = json.loads(content)
queries = parsed.get("queries", [])
```

第 157-173 行：清理 queries。

它會：

- 跳過不是 dict 的 item。
- 清掉多餘空白。
- 跳過空 query。
- 跳過重複 query。
- intent 空的話補成 `"places_search"`。
- 最多回傳 `max_queries` 筆。

第 174-177 行：任何錯誤都 fallback。

```python
except Exception:
    pass

return _fallback_queries(...)
```

這讓流程不容易中斷，但缺點是錯誤被吞掉，debug 時比較難知道是 OpenAI 失敗還是 JSON 格式錯。

建議：至少把錯誤放進 debug metadata。

### 第 180-248 行：GooglePlacesTextSearch class

這個 class 封裝 Google Places Text Search API。

#### 第 181-197 行：初始化

```python
class GooglePlacesTextSearch:
    def __init__(..., api_key=None, include_quality_fields=None, timeout_seconds=15):
```

第 187-191 行：

```python
self.api_key = (
    api_key
    or os.getenv("GOOGLE_PLACES_API_KEY")
    or os.getenv("GOOGLE_MAPS_API_KEY")
)
```

API key 優先順序：

1. 函式參數傳入。
2. `.env` 的 `GOOGLE_PLACES_API_KEY`。
3. `.env` 的 `GOOGLE_MAPS_API_KEY`。

第 192-196 行：

```python
self.include_quality_fields = (
    _truthy_env("GOOGLE_PLACES_INCLUDE_QUALITY_FIELDS")
    if include_quality_fields is None
    else include_quality_fields
)
```

如果呼叫者沒指定，就看 `.env`。如果呼叫者指定，就用指定值。

例子：

```python
GooglePlacesTextSearch(include_quality_fields=False)
```

只拿基本欄位，省成本。

```python
GooglePlacesTextSearch(include_quality_fields=True)
```

多拿評分、營業時間等欄位。

第 197 行：

```python
self.timeout_seconds = timeout_seconds
```

HTTP timeout 預設 15 秒。

#### 第 199-203 行：組 field mask

```python
def _field_mask(self) -> str:
    fields = list(BASE_FIELD_MASK)
    if self.include_quality_fields:
        fields.extend(QUALITY_FIELD_MASK)
    return ",".join(fields)
```

如果 `include_quality_fields=False`：

```text
places.id,places.name,places.displayName,...
```

如果是 True，就加上：

```text
places.rating,places.userRatingCount,places.regularOpeningHours,...
```

#### 第 205-248 行：呼叫 Google Places Text Search

```python
def search_text(query, page_size=6, language_code="zh-TW", region_code="TW", location_bias=None)
```

第 213-217 行：沒有 API key 就丟錯。

```python
if not self.api_key:
    raise RuntimeError(...)
```

第 219 行：

```python
safe_page_size = max(1, min(int(page_size), 20))
```

Google Places page size 最多限制成 20。  
如果傳 0，會變 1；如果傳 100，會變 20。

第 220-225 行：組 request body。

```python
body = {
    "textQuery": query,
    "pageSize": safe_page_size,
    "languageCode": language_code,
    "regionCode": region_code,
}
```

例子：

```json
{
  "textQuery": "台南 國華街 牛肉湯",
  "pageSize": 6,
  "languageCode": "zh-TW",
  "regionCode": "TW"
}
```

第 226-227 行：如果有 location bias 就加進 body。

例子：

```json
{
  "locationBias": {
    "circle": {
      "center": {"latitude": 22.9974, "longitude": 120.2025},
      "radius": 1500
    }
  }
}
```

目前上層沒有傳，所以這個能力還沒被使用。

第 229-238 行：發 HTTP POST。

```python
response = requests.post(
    TEXT_SEARCH_URL,
    headers={...},
    json=body,
    timeout=self.timeout_seconds,
)
```

headers 內：

- `Content-Type`: JSON。
- `X-Goog-Api-Key`: API key。
- `X-Goog-FieldMask`: 欄位清單。

第 240-244 行：Google 回 400 以上就丟 RuntimeError。

例子：

```text
403: API key 沒開 Places API
429: 超出 quota
400: request body 格式錯
```

第 246-247 行：成功就 parse JSON，回傳 `places` list。

### 第 250-263 行：Google place 名稱與 ID helper

```python
def _display_name(place):
```

Google 的 displayName 可能長這樣：

```json
{"text": "國華牛肉湯", "languageCode": "zh-TW"}
```

這個 helper 取出 `"國華牛肉湯"`。

```python
def _place_id(place):
```

優先取 `place["id"]`。  
如果沒有，就從 resource name `places/ChIJ...` 裡切出 `ChIJ...`。

例子：

```text
places/ChIJ18NJrmZ2bjQRLm4dURTcDtQ
→ ChIJ18NJrmZ2bjQRLm4dURTcDtQ
```

### 第 266-307 行：normalize Google place

```python
def _normalize_place(place, query, query_intent, rank_index):
```

這是把 Google 原始格式轉成我們內部候選格式。

第 272-275 行：

```python
location = place.get("location") or {}
rating = place.get("rating")
user_rating_count = place.get("userRatingCount")
google_rank_score = round(1.0 / (rank_index + 1), 4)
```

`google_rank_score` 是簡單 rank 分：

```text
第 1 名 = 1.0
第 2 名 = 0.5
第 3 名 = 0.3333
第 6 名 = 0.1667
```

第 277-291 行：建立 normalized dict。

例子：

```json
{
  "place_id": "ChIJ...",
  "resource_name": "places/ChIJ...",
  "name": "國華牛肉湯",
  "address": "70055臺南市中西區...",
  "latitude": 22.991037,
  "longitude": 120.1969604,
  "primary_type": "restaurant",
  "types": ["restaurant", "food", "point_of_interest"],
  "google_maps_uri": "https://maps.google.com/...",
  "matched_queries": [
    {"query": "台南 國華街 牛肉湯", "intent": "restaurant", "rank": 1}
  ],
  "ranking_signals": {
    "google_rank_score": 1.0
  }
}
```

第 293-305 行：如果 Google 有回 quality 欄位，就加進 normalized。

例如：

```python
if rating is not None:
    normalized["rating"] = rating
```

注意：如果 `include_quality_fields=False`，這些欄位通常不會存在，所以後面的 scorer 會用中立分或 warning 處理。

### 第 310-327 行：合併重複候選

```python
def _merge_candidates(candidates):
```

同一個地點可能被多個 query 命中。

例子：

```text
query 1: 台南 國華街 牛肉湯 → 國華牛肉湯
query 2: 台南 國華街 午餐 → 國華牛肉湯
```

這時不應該有兩個「國華牛肉湯」，而是合併成一個候選，並保留兩個 matched_queries。

第 311-312 行：準備 dict 與順序 list。

第 314-315 行：用 place_id 當 key；如果沒有 place_id，就用 `name|address`。

第 316-319 行：第一次看到這個 key，就加入。

第 321-325 行：如果已存在：

- 把新的 matched_queries 合併進去。
- google rank score 取最高。

例子：

```json
"matched_queries": [
  {"query": "台南 國華街 牛肉湯", "rank": 2},
  {"query": "台南 國華街 午餐", "rank": 1}
]
```

第 327 行：照第一次出現順序回傳。

### 第 330-380 行：主流程 search_places_for_pending_searches

```python
def search_places_for_pending_searches(...):
```

這是外部主要呼叫的函式。

參數：

- `pending_searches_json`：Extractor 輸出的模糊 slot。
- `destination_hint`：目的地，例如台南。
- `max_queries_per_slot`：每個 slot 產生幾個 query，預設 4。
- `max_results_per_query`：每個 query 拿幾筆 Places 結果，預設 6。
- `include_quality_fields`：是否拿評分/營業時間等欄位。
- `expand_queries_with_llm`：是否用 LLM 產生多組 query。

第 338-340 行：

```python
pending_searches = _extract_pending_searches(pending_searches_json)
searcher = GooglePlacesTextSearch(include_quality_fields=include_quality_fields)
slots = []
```

整理 pending searches，建立 Google Places searcher。

第 342 行：

```python
for task in pending_searches:
```

對每個模糊 slot 搜尋候選。

第 343-350 行：產生 queries。

```python
if expand_queries_with_llm:
    queries = generate_places_queries(...)
else:
    queries = _fallback_queries(...)
```

例子：

```text
slot = 安平老街午餐（豆花、蝦捲等）
queries =
- 台南 安平老街 豆花
- 台南 安平老街 蝦捲
- 台南 安平老街 午餐
```

第 352-353 行：準備候選與錯誤容器。

第 354-362 行：每個 query 呼叫 Google Places。

```python
places = searcher.search_text(query, page_size=max_results_per_query)
for index, place in enumerate(places):
    raw_candidates.append(_normalize_place(place, query, intent, index))
```

如果某個 query 失敗：

```python
errors.append({"query": query, "error": str(exc)})
```

這代表整個 slot 不會因一個 query 失敗就中斷。

第 364-378 行：組成 candidate slot。

輸出包含：

```json
{
  "slot_id": "day1_slot05",
  "day": 1,
  "start_time": "12:10",
  "end_time": "13:30",
  "time": "12:10~13:30",
  "slot_type": "meal",
  "original_activity": "在國華街享用午餐...",
  "reason_for_search": "缺乏具體店家資訊",
  "queries": [...],
  "candidates": [...],
  "errors": []
}
```

第 380 行：

```python
return {"candidate_slots": slots}
```

這就是 CandidateScorer 的輸入。

### 第 383-408 行：CLI 測試入口

```python
def main() -> None:
```

讓你可以直接執行：

```bash
python corporate_with_gemini/PlacesSearchAgent.py < pending.json
```

第 384 行：讀 stdin。

第 385-386 行：如果 stdin 有資料，就用它。

第 387-398 行：如果沒 stdin，就用一個測試 payload。

例子：

```json
{
  "day": 1,
  "time": "10:00~11:00",
  "original_activity": "享用台南特色牛肉湯作為早午餐",
  "reason_for_search": "缺乏具體店家",
  "suggested_search_query": "台南 必吃 牛肉湯 早午餐"
}
```

第 400-403 行：呼叫主搜尋流程。

```python
destination_hint=os.getenv("TRAVEL_DESTINATION_HINT", "台南")
```

這裡還有台南 fallback。它只影響單獨跑這支檔案的 demo，不影響 `subMain.py` 的主流程，但仍建議之後拿掉。

第 404 行：印出 JSON 結果。

第 407-408 行：

```python
if __name__ == "__main__":
    main()
```

只有直接執行這個檔案時才跑 `main()`。如果被別的檔案 import，不會自動搜尋。

---

## 總結

這支程式目前的責任切得算清楚：

```text
模糊 slot → 多組 Places query → Google Places → 候選池
```

主要 hard-coding 不在「食物字典」那種層級，而是在：

- 固定 Google Places provider。
- 固定 `zh-TW` / `TW`。
- prompt 範例偏台南。
- fallback query 很簡單。
- 每 slot query/results 數量固定。
- field mask profile 還沒有正式配置化。
- 尚未使用 anchor location bias。
- CLI demo 還有台南 fallback。

最值得下一步改的是：

```text
讓 PlacesSearchAgent 接收 trip_context + anchor_context，
用 destination country 決定 region_code，
用前後 anchor 中心點做 location_bias，
並把 provider 抽象化，未來才能替換 OSM/Geoapify。
```
