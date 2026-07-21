# TravelEval Partial Metrics Without Gaode

這不是官方完整 TravelEval 分數；它輸出 paper-style metric table，去除高德 API 必要項目。

↑ 表示越高越好，↓ 表示越低越好，- 表示成本分解值本身不代表好壞。

## Computed Metrics

| Dimension | Metric | Direction | Value | Status | Note |
|---|---:|:---:|---:|---|---|
| Accuracy | CCD | ↓ | 0.0000 | computed_from_converted_plan | 成本表由轉換器重算；可抓 arithmetic mismatch，但不是模型自報成本。 |
| Accuracy | FAR | ↓ | 0.2000 | computed_partial | 只檢查 attraction；可由 TravelEval sandbox 或候選座標佐證。 |
| Accuracy | VROH | ↓ | 0.0000 | computed_partial | 只檢查能匹配到 TravelEval attractions CSV 的景點。 |
| Accuracy | ITD | ↓ | NA | missing_input | final result 沒有火車/航班班次與時間，無法用 TravelEval intercity sandbox 算 ITD。 |
| Accuracy | PD | ↓ | 0.0000 | computed |  |
| Compliance | BCS | ↑ | 1.0000 | computed | 成本與預算統一以 CNY 計。 |
| Compliance | TCS | ↑ | 1.0000 | computed_partial | 檢查天數與時間格式/重疊；未檢查實際交通可行時間。 |
| Compliance | HCS | ↑ | 1.0000 | computed_partial |  |
| Compliance | PAS-A | ↑ | 0.7500 | computed_partial | 無住宿偏好時，夜數正確即視為滿足。 |
| Compliance | PAS-T | ↑ | 1.0000 | proxy_no_gaode | 不查實際路線，只看偏好交通是否出現在行程/通勤 slot。 |
| Compliance | PAS-C | ↑ | 1.0000 | proxy_text_match | 用飲食偏好文字覆蓋近似。 |
| Compliance | TTCS | ↑ | 0.0000 | computed_partial | 使用 TravelEval 推薦停留時間；未納入排隊時間模型。 |
| Temporality | STR | ↑ | 0.6667 | proxy_no_queue_model | 用停留時間落在推薦區間的程度近似，未納入排隊時間。 |
| Temporality | DTU | ↑ | 0.3333 | proxy_no_queue_model | 用每日景點時數 / 當日行程 span 近似。 |
| Temporality | OTU | ↑ | 0.1250 | proxy_no_queue_model | 用總景點時數 / 24h*天數近似。 |
| Economy | BE | ↑ | 7.2056 | proxy_estimated_costs | TravelEval 公式方向：experience value / actual cost * 1000；目前成本含住宿、景點票、餐費與市內交通估算，仍缺城際交通。 |
| Economy | ACD | - | 444.6000 | estimated_or_computed | 住宿成本。 |
| Economy | ATD | - | 380.0000 | estimated_or_computed | 景點票價優先來自 slot cost_estimate，缺少時用 TravelEval attraction price fallback。 |
| Economy | ETD | - | 0.0000 | missing_or_zero | 城際交通成本目前缺資料。 |
| Economy | RTD | - | 2000.0000 | estimated_or_computed | 市內交通成本優先來自 slot cost_estimate，缺少時用每段整團 8 CNY fallback。 |
| Economy | MED | - | 520.0000 | estimated_or_computed | 餐費優先來自 slot cost_estimate，缺少時用 TravelEval restaurant price 或每人 80 CNY fallback。 |
| Economy | OTD | - | 0.0000 | missing_or_zero | 其他成本。 |
| Utility | EDI | ↑ | 0.5749 | computed_partial | 用 TravelEval attraction type 做 Shannon entropy。 |
| Utility | ADS | ↑ | 0.3531 | computed_partial | 平均每日景點數 / 普通旅行 target density 3.54。 |
| Utility | AQE | ↑ | 0.9600 | computed_partial | 用景點 star/5 加權停留時間近似。 |
| Utility | Profit | ↑ | 6.0250 | computed_partial | 用 sandbox star 與偏好匹配近似。 |

## Skipped Gaode Metrics

| Dimension | Metric | Direction | Reason |
|---|---:|:---:|---|
| Accuracy | FRTC | ↓ | 需要高德 transit API 檢查相鄰 POI 公共交通可達性。 |
| Spatiality | SSR | ↓ | 需要高德路網距離計算 route penalty。 |
| Spatiality | CSM-P90 | ↓ | 需要 POI 路網距離與跨日空間錯配計算。 |
| Spatiality | CSM-P95 | ↓ | 需要 POI 路網距離與跨日空間錯配計算。 |
