# Evaluation 說明

這個資料夾放 TravelEval 相關的轉換與離線評分工具，避免混在主 pipeline 裡。

## 新增檔案

### `traveleval_offline_adapter.py`

用途：

1. 讀取 `FinalItineraryAgent.py` 產出的 final result。
2. 讀取 `PlannerAgent.py` 的原始 result，拿回 5W1H 條件，例如目的地、天數、人數、預算、飲食偏好、key areas。
3. 把 final result 轉成 TravelEval 原始程式期待的格式：
   - `summary`
   - `intercity_transport`
   - `accommodation`
   - `daily_plans`
   - `cost_breakdown`
4. 跳過需要高德 API 的評分項目。
5. 用其他可離線計算的項目打一個平均分。

預設輸出會放在：

```text
corporate_with_gemini/Evaluation/outputs/
```

會產生兩個檔案：

```text
*_traveleval_plan.json
*_offline_score_without_gaode.json
*_partial_metrics_without_gaode.json
*_partial_metrics_without_gaode.md
```

第一個是轉成 TravelEval schema 的行程，第二個是離線 summary score。

後兩個是更建議用來和 paper 對照的 partial metrics table。它們不報單一總分，而是用 paper 類似的欄位列出 `CCD/FAR/VROH/BCS/TTCS/STR/BE/EDI/ADS/...`，並把需要高德 API 的 `FRTC/SSR/CSM` 放到 skipped 區。

## 怎麼直接測試

直接執行：

```bash
python3 corporate_with_gemini/Evaluation/traveleval_offline_adapter.py
```

預設會讀：

```text
corporate_with_gemini/traced_outputs/北京上海測資_result_candidate_llm_input_final_itinerary.json
corporate_with_gemini/traced_outputs/北京上海測資_result.json
```

如果要改預設檔名，可以改 `traveleval_offline_adapter.py` 最上方這兩行：

```python
DEFAULT_FINAL_FILENAME = "你的_final_itinerary.json"
DEFAULT_PLANNER_FILENAME = "你的_planner_result.json"
```

也可以用指令指定：

```bash
python3 corporate_with_gemini/Evaluation/traveleval_offline_adapter.py final.json planner.json output_prefix
```

## 這版跟 TravelEval 原始評分的關係

我有先看 `/private/tmp/TravelEval/` 的原始碼。TravelEval 的主要入口是：

```text
/private/tmp/TravelEval/core/evaluator.py
```

它會建立六個維度：

```text
accuracy
constraint
time
space
economy
utility
```

而 `PlanExtractor` 期待 AI plan 裡有 `summary`、`intercity_transport`、`accommodation`、`daily_plans`、`cost_breakdown`。所以這版 adapter 不是只把行程轉成純文字，而是轉成比較接近 TravelEval 原始 schema 的 JSON。

## 這版跳過哪些高德相關評分

目前先跳過：

1. `space` 整個維度
   - TravelEval 的 `SpaceMetrics` 會用 `GeoCalculator` 算路線懲罰 `RP` 和跨日空間錯配 `CSM`。
   - 這部分會碰到高德路線/距離能力。

2. `accuracy.transportation_breaks`
   - TravelEval 會檢查相鄰 POI 之間公共交通是否可達。
   - 這也會碰到高德路線能力。

這些項目不會被算進 `overall_score_without_gaode`。

## 這版分數怎麼解讀

輸出的離線 summary score 是：

```text
overall_score_without_gaode
```

範圍是 `0.0 ~ 1.0`。

它是把可離線評分的維度取平均，不包含 `space`，也不包含需要高德路線 API 的交通斷裂檢查。

重要限制：

1. 這不是 TravelEval 官方完整分數。
2. 它是「TravelEval schema 對齊 + 無高德 API 的離線近似分數」，不應直接拿去和 paper 表格比較。
3. 目前測資預算先視為 `CNY`，SerpApi 住宿若是 `TWD`，轉換器會先用固定公式 `CNY = TWD / 5` 換算後再評分。
4. 轉換器會讀 final result 裡每個 slot 的 `cost_estimate`；如果 slot 沒有價格，會先用 TravelEval attraction/restaurant price 當 fallback。
5. 市內交通若沒有價格，暫以每段整團 `8 CNY` 粗估；餐費若無法匹配 TravelEval restaurant，暫以每人 `80 CNY` 粗估。
6. 因為 final result 目前仍沒有完整城際交通票價，`economy` 和 `BCS` 仍是 partial/estimated，不是官方完整 TravelEval 成本評估。

如果要跟 TravelEval paper 對照，請優先看：

```text
*_partial_metrics_without_gaode.md
```

這份表格保留 metric 原本的方向：

```text
↑ 越高越好
↓ 越低越好
- 成本分解值，不直接代表好壞
```

## 建議的 slot 成本格式

之後可以讓最後定案的 LLM 對每個 slot 補一個 `cost_estimate`：

```json
{
  "slot_id": "day3_slot01",
  "cost_estimate": {
    "currency": "CNY",
    "per_person": 399,
    "party_total": 798,
    "source": "llm_estimated",
    "confidence": 0.65,
    "note": "上海迪士尼門票粗估"
  }
}
```

轉換器會優先讀 `party_total`，沒有的話讀 `total` 或 `amount`，再沒有才用 `per_person * traveler_count`。

## 高德 API 怎麼用比較安全

如果之後真的要補完整 TravelEval 空間分數，建議這樣做：

1. API key 只放在環境變數，例如 `AMAP_API_KEY`，不要寫進 code 或 JSON。
2. `.env` 要加入 `.gitignore`，不要 commit。
3. 只在後端讀 key，不要讓前端或輸出 JSON 帶出 key。
4. log 裡不要印完整 request URL，因為 URL query 可能含 key。
5. 先做 route cache，同一段路不要重複打 API。
6. 在高德平台設定配額、監控、告警；如果支援 IP 或網域限制，也打開。
7. 定期輪替 key。若不小心輸出或 commit，立刻 revoke。
