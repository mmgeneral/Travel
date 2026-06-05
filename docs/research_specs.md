1. 攔截器與快慢路徑 (Fast-path Routing)
參考文獻： 2603.01548 (Self-Healing Router) 核心邏輯： 透過「平行健康監測器 (Parallel Health Monitors)」與「確定性圖導航」取代昂貴的 LLM 推理。對於例行性決策（如：Stripe 壞了換 Razorpay），系統直接在圖上重繪權重並執行，不進入 LLM
。
實作清單：
平行監測器 (PHM)： 實作輕量級的 Regex 或小模型監測器，對 Intent (意圖)、Risk (風險) 與 Tool Health (工具狀態) 進行評分
。
成本權重圖 (Cost-weighted Tool Graph)： 將旅遊工具（機票、飯店、支付）定義為節點，邊的權重代表成本
。
Dijkstra 路徑計算： 當監測器偵測到故障（權重變無限大）時，自動計算最短路徑
。
虛擬碼：
def fast_path_orchestrator(user_goal, tool_graph):
    # 1. 平行執行監測器，不消耗 LLM Token
    signals = parallel_health_monitors.run_all(context)
    winner = max(signals, key=lambda x: x.priority)
    
    # 2. 如果是已知工具故障或風險，更新圖權重
    if winner.priority > 0.9:  # 例如：訂票 API 回傳 500
        tool_graph.update_edge_weight(winner.target_node, float('inf'))
    
    # 3. 執行確定性路徑搜尋 (Fast-path)
    path = dijkstra_shortest_path(tool_graph, start_node, user_goal)
    
    if path:
        return execute_path(path)  # 成功跳過 LLM
    else:
        return escalate_to_llm(user_goal)  # 僅在無路可走時調用 LLM (Slow-path) [8, 9]

--------------------------------------------------------------------------------
2. 結構化通訊協議 (Structured Communication)

### 2.1 外部通訊：客戶端與 Agent 串流協議 (Client-Server)

2.1.1 核心設計原則
單一資料源 (Single Source of Truth)： 所有 Agent 的內部狀態與思考過程，皆應透過標準化 JSON 格式輸出，前端僅負責渲染，不包含任何業務邏輯。

漸進式狀態更新 (Progressive State Updates)： Agent 在執行 Retriever, Critic, Planner 等節點時，需即時廣播當前進度，以減少使用者的等待焦慮。

明確的行動邊界 (Actionable Boundaries)： 嚴格區分「可執行 (Actionable)」與「需釐清 (Needs Clarification)」的意圖，強制中斷並要求使用者補齊資訊。

2.1.2 客戶端請求結構 (Client Request Payload)
前端發起規劃請求時，需夾帶使用者的原始輸入、地理位置及語系，以便後端進行多語言或在地化處理。

[POST] /agent/query

JSON
{
  "query": "京都 TASTE_MAX",
  "context": {
    "user_locale": "zh_TW",
    "user_lat": 24.8138,
    "user_lng": 120.9675,
    "session_id": "sess_abc123",
    "previous_intent": null
  }
}
2.1.3 伺服端串流事件 (Server Event Stream)
後端接收請求後，會持續吐出多個 Event 直到規劃完成或需要追問。前端需根據 event 類型進行不同的 UI 更新。

事件類型列表：
status_update: 更新 UI 上的進度指示器（例如：• 正在搜尋候選餐廳）。

debug_print: 開發者除錯用的後台 Log（例如：Node critic starting）。

clarify_required: 意圖解析不完整，要求前端彈出選項讓使用者選擇。

final_itinerary: 最終行程生成完畢。

範例 Payload (SSE 格式)：
情境 A：正常規劃流程

JSON
// Event 1: 解析意圖
{"event": "status_update", "message": "正在解析您的旅遊意圖"}

// Event 2: 搜尋與審查
{"event": "status_update", "message": "正在搜尋候選餐廳"}
{"event": "debug_print", "node": "node_retriever", "message": "RetrieverAgent starting"}

// Event 3: 規劃完成，回傳最終行程
{"event": "final_itinerary", "data": {
  "day": 1,
  "city": "京都",
  "mode": "taste_max",
  "shops": [
    {"time": "11:30 - 13:00", "name": "肉のはせ川", "note": "Walk"}
  ]
}}
情境 B：意圖不明，觸發追問 (Fast-path 攔截或 LLM 判定)
當解析出的 Intent 物件中 is_actionable 為 false 時，後端將中斷流程並發送追問事件。

JSON
{"event": "status_update", "message": "正在確認您的需求..."}
{"event": "clarify_required", "data": {
  "type": "location_missing",
  "question": "請指定要規劃的城市（無法從您的描述推斷）：\n(A) 東京\n(B) 大阪\n(C) 京都\n(D) 台北",
  "options": ["A", "B", "C", "D"]
}}
(備註：前端收到 clarify_required 後，應呼叫 showClarifyPanel(data) 渲染按鈕。)

2.1.4 Intent 資料結構 (Data Model)
此為系統內部在各 Node 之間傳遞，以及最終可能同步給前端的核心物件定義。

JSON
{
  "city": "京都",                  // 字串或 null
  "region": "jp",                  // 國家/區域代碼
  "meal_slots": [],                // 預計用餐時段 (breakfast, lunch, etc.)
  "time_window": [null, null],     // 特殊時間限制
  "mode": "taste_max",             // 偏好模式 (balanced, taste_max 等)
  "must_include_shops": [],        // 強制排入的店家
  "must_exclude_shops": [],        // 黑名單店家
  "is_actionable": true,           // 是否可直接進入規劃階段
  "actionability_followup": null   // 若不可執行，存放給使用者的追問文字
}


### 2.2 內部通訊：階層式升級協議 (Agent-to-Agent Escalation Protocol)
參考文獻： 2604.11378 (Graph Harness / SGH) 核心邏輯： 規範 Researcher (執行層) 與 Critic (修復/檢驗層) 的互動。透過「階層式升級協議 (Escalation Protocol)」防止 Agent 陷入無限重試或規劃迴圈
。
實作清單：
JSON 回傳格式規範： Researcher 必須回傳滿足「輸出契約 (Contract)」的結果
。
三級升級協議： 依序為 Local Retry (重試) -> Local Patch (局部修補) -> Request Replan (全局重規劃)
。
最大循環次數 (Max Iteration)： 設定 retry_budget 與 replan_limit
。
結構化 JSON 格式範例：
{
  "node_id": "hotel_booking_001",
  "status": "failed",
  "error_type": "contract_violation", // 契約違反
  "diagnostics": {
    "reason": "Missing breakfast selection",
    "suggested_action": "local_patch" // 建議動作 [16]
  },
  "context_isolated": true // 確保執行與診斷上下文分離 [17]
}
虛擬碼：
def escalation_protocol(node_id, failure_report):
    # 定義最大循環次數 [18]
    MAX_RETRY = 3
    MAX_PATCH = 2

    # L1: 局部重試 (Local Retry)
    if recovery_state[node_id].retry_count < MAX_RETRY:
        return action_retry(node_id)
        
    # L2: 局部修補 (Local Patch - 修改節點配置而不變動圖結構)
    elif recovery_state[node_id].patch_count < MAX_PATCH:
        return action_patch(node_id, failure_report.suggested_cfg)
        
    # L3: 請求重規劃 (Request Replan - 生成新版本的 DAG)
    else:
        return request_new_plan_version(reason=failure_report.reason) # [11, 19]

--------------------------------------------------------------------------------
3. 意圖補全與追問機制 (Self-Correcting Clarification)
參考文獻： 2025.emnlp-main.682 (ASKTOACT) 核心邏輯： 當使用者查詢（如「訂機票」）缺少關鍵參數（如「出發地」）時，系統不應猜測，而是根據工具所需的參數（Tool Parameters）主動追問，並提供選項
。
實作清單：
參數分析 (Parameter Analysis)： 將 User Query 與 API Required Parameters 比對，找出缺失值
。
選項生成機制： 參考歷史數據或 API 限制（如：列出前三熱門出發地），提供使用者點選
。
自我修正 (Self-Correction)： 若 Agent 追問了「使用者已提供過」的資訊，需觸發自我糾正語句（如：抱歉，我漏看了，您已經說過是在台北出發...）
。
虛擬碼：
def self_correcting_clarification(user_query, target_api):
    # 1. 識別缺失參數 [27, 28]
    missing_params = analyze_missing_parameters(user_query, target_api)
    
    for param in missing_params:
        # 2. 執行錯誤偵測 (是否為冗餘追問？) [25, 29]
        if is_clearly_stated_in_context(param, user_query):
            trigger_self_correction(param) # "抱歉，我發現您已經提到過..."
            continue
            
        # 3. 生成具備選項的追問問題 (qc) [30, 31]
        options = api_registry.get_parameter_options(param)
        question = f"請問您的 {param} 是什麼？您可以選擇：{', '.join(options)}"
        
        # 4. 暫停規劃，向使用者點選/輸入 [32]
        user_response = await_user_interaction(question, options)
        update_transformation_record(param, user_response) # [24, 33]