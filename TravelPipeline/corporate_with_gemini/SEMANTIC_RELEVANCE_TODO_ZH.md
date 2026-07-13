# 待解決問題：CandidateScorer 語意相關度

## 問題摘要

目前 `CandidateScorer.py` 的 relevance score 已經改成優先使用 `target_terms`、`location_terms`、`intent_type`、`must_have` 等結構化欄位，但最後的比對方式仍然主要是字串匹配與中文單字重疊。

這代表它比單純切查詢字串穩定，但還不是完整語意理解。

## 目前做法

主路徑：

1. `Mcp.py` 從模糊 slot 拆出 `target_terms`、`location_terms`、`intent_type`、`must_have`。
2. `PlacesSearchAgent.py` 的 query generator 也會為每組 Google Places query 輸出同樣的結構化欄位。
3. `CandidateScorer.py` 優先用這些欄位計算 relevance。
4. 如果 structured terms 不存在，才退回 heuristic 字串切詞。

## 目前限制

- 不知道「語意相近但字面不同」的情況，例如「老屋咖啡」與「町家喫茶」。
- 中文字元重疊可能造成誤判，例如「文化」可能同時出現在景點、路名、停車場。
- Google Places type 與 query intent 的比對仍是文字層級，不是 ontology 或 embedding。
- 還不能根據使用者偏好動態判斷「這個候選是否真的符合期待」。

## 後續解法候選

### 方案 A：Embedding reranker

把 slot 的 structured intent 和 candidate 的 name/type/address 組成文字，計算 embedding similarity。

優點：

- 比字串匹配更懂語意。
- 可以保留 deterministic scoring pipeline。

缺點：

- 需要多一次 embedding API 或本地 embedding 模型。
- 要設計 similarity threshold。

### 方案 B：LLM relevance judge

讓小模型判斷 candidate 是否符合 slot，輸出 0 到 1 的 relevance score 與理由。

優點：

- 最符合目前「盡可能交給 LLM，少寫規則」的方向。
- 可以理解複雜需求，例如「適合雨天」「適合帶長輩」「不要太商業化」。

缺點：

- token 成本較高。
- 需要嚴格 JSON schema 與快取，不然容易不穩。

### 方案 C：混合式

先用現在的 heuristic scorer 篩到每格前 5 名，再用 embedding 或 LLM judge rerank 成前 2 名。

這是我目前最推薦的論文 prototype 方向，因為它兼顧成本、速度、可解釋性與語意彈性。

## 論文描述方式

可以把目前版本稱為：

> structured-term heuristic relevance scorer

後續改良版可以稱為：

> semantic reranking module

比較實驗可以設計成：

1. 只用 Google Places rank。
2. Google rank + heuristic scorer。
3. Google rank + heuristic scorer + semantic reranker。

觀察指標可以是人工偏好評分、是否符合 slot 意圖、是否減少不相關候選、是否降低 LLM refinement token。
