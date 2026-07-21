from __future__ import annotations

import copy


def verify_planner_slots(planner_result: dict, raw_constraints: dict) -> dict:
    """
    檢查 planner_result 裡每個 slot_type=="meal" 且 is_searchable==False 的項目，
    如果 raw_constraints 非空，強制設成 searchable True。
    """
    constraints = raw_constraints.get("raw_constraints", {})
    if not constraints:
        # 沒有約束就回傳原樣
        return planner_result

    modified_ids = []
    for day_obj in planner_result.get("itinerary", []):
        if not isinstance(day_obj, dict):
            continue
        schedule = day_obj.get("schedule", [])
        if not isinstance(schedule, list):
            continue
        for item in schedule:
            if not isinstance(item, dict):
                continue
            if item.get("slot_type") != "meal":
                continue
            if item.get("is_searchable") is False or item.get("is_searchable") is None:
                # 強制改成 True
                item["is_searchable"] = True
                slot_id = item.get("slot_id")
                if slot_id:
                    modified_ids.append(slot_id)

    if modified_ids:
        print(
            f"[Verifier] 主 Planner 以下 slot 被改為可搜尋（因使用者約束）：{modified_ids}"
        )

    return planner_result
