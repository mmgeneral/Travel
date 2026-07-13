# Pipeline Output 編號建議

## 建議結論

建議輸出檔名按照「實際執行順序」命名，而不是按照最初論文流程的抽象步驟命名。

目前我建議：

| 編號 | 模組 | 輸出檔 |
|---|---|---|
| 00 | Run metadata | `00_run_metadata.json` |
| 01 | Initial Planner LLM | `01_planner_agent_output.json` |
| 02 | Ambiguity Extractor | `02_extract_searchable_activities_output.json` |
| 02 raw | Ambiguity Extractor 原始文字 | `02_extract_searchable_activities_output.raw.json` |
| 03 | Anchor Resolver | `03_anchor_context_output.json` |
| 04 | Search Topic Generator + Places Candidate Search | `04_places_search_agent_output.json` |
| 05 | Candidate Scorer 完整 debug | `05_candidate_score_output.json` |
| 05 shortlist | 給 Internal Refinement LLM 的前兩名候選 | `05_candidate_shortlist_for_refinement.json` |
| 05 archive | 本次未入選候選 | `05_candidate_archive.jsonl` |
| 06 | Internal Refinement LLM | `06_internal_refinement_output.json` |
| 07 | Rule Validator | `07_rule_validator_output.json` |
| 08 | User-facing Draft | `08_user_facing_draft.json` |

## 為什麼 Anchor Resolver 放在 03

雖然論文主流程裡 Candidate Search 是第四步，但目前實作上 Places Search 會使用 anchor context 做 location bias。

所以實際順序是：

1. Planner 先產生完整時間軸。
2. Extractor 找出模糊 slot。
3. AnchorResolver 找出模糊 slot 前後站。
4. PlacesSearchAgent 用 anchor context 做 location bias，搜尋候選池。
5. CandidateScorer 用 anchor context 算距離分。

因此 `03_anchor_context_output.json` 比 `04_places_search_agent_output.json` 更合理。

## Search Topic Generator 要不要獨立成 03

現在 Search Topic Generator 還包在 `PlacesSearchAgent.py` 裡，因此它的 output 會保存在 `04_places_search_agent_output.json` 的每個 slot 的 `queries` 欄位中。

如果之後要更細緻追蹤 token 或重跑，可以拆成：

- `03_search_topic_generator_output.json`
- `04_anchor_context_output.json`
- `05_places_search_agent_output.json`
- `06_candidate_score_output.json`

但目前 prototype 階段，我建議先不要拆太細，避免流程文件比程式本身還難讀。
