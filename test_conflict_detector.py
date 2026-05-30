import pytest
from agent import _conflict_check_travel

# 測試用的 mock catalog
# 兩家店距離遠（約 10 km，travel_min ≈ 20 分鐘）
# 兩家店距離近（約 1 km，travel_min ≈ 5 分鐘，MIN_TRAVEL）

CATALOG_FAR = {
    "店A": {"latitude": 35.0, "longitude": 135.7},
    "店B": {"latitude": 35.09, "longitude": 135.7},   # ~10 km
    "店C": {"latitude": 35.18, "longitude": 135.7},   # ~10 km from B
}

CATALOG_NEAR = {
    "店A": {"latitude": 35.0, "longitude": 135.7},
    "店B": {"latitude": 35.001, "longitude": 135.7},  # ~0.1 km
    "店C": {"latitude": 35.002, "longitude": 135.7},
}

def make_slot(slot_id, shop_name, start_time, duration_minutes=90):
    return {
        "slot_id": slot_id,
        "shop_name": shop_name,
        "start_time": start_time,
        "duration_minutes": duration_minutes,
        "meal_type": "lunch",
        "user_locked": False,
        "session_locked": False,
    }

# ─── Case 1: 無衝突，店家很近 ───
def test_no_conflict_when_shops_are_near():
    slots = [
        make_slot("s1", "店A", "09:00"),
        make_slot("s2", "店B", "10:45"),  # 09:00 + 90min + 15min buffer
        make_slot("s3", "店C", "12:30"),
    ]
    result = _conflict_check_travel(slots, "s2", CATALOG_NEAR)
    assert result is None

# ─── Case 2: 前一個 slot 來不及 ───
def test_conflict_from_prev_slot():
    slots = [
        make_slot("s1", "店A", "09:00", duration_minutes=90),
        make_slot("s2", "店B", "10:00"),  # 只有 30 min，但需要 ~20 min travel（遠）
    ]
    result = _conflict_check_travel(slots, "s2", CATALOG_FAR)
    # 距離遠 travel_min > 0 min available → 衝突
    assert result is not None
    assert result["conflict_type"] == "transport"
    assert "店A" in result["message"]

# ─── Case 3: 到下一個 slot 來不及 ───
def test_conflict_to_next_slot():
    slots = [
        make_slot("s1", "店A", "09:00", duration_minutes=90),
        make_slot("s2", "店B", "10:30", duration_minutes=90),
        make_slot("s3", "店C", "11:00"),  # s2 結束 12:00，只有 -60 min
    ]
    result = _conflict_check_travel(slots, "s2", CATALOG_FAR)
    assert result is not None
    assert "店C" in result["message"]

# ─── Case 4: 兩邊衝突，有可行時間窗口 ───
def test_conflict_both_sides_with_feasible_window():
    slots = [
        make_slot("s1", "店A", "09:00", duration_minutes=60),
        make_slot("s2", "店B", "09:30", duration_minutes=60),  # 來不及（需要 ~20 min）
        make_slot("s3", "店C", "14:00", duration_minutes=60),
    ]
    result = _conflict_check_travel(slots, "s2", CATALOG_FAR)
    assert result is not None
    assert result["feasible_window"] is not None
    fw = result["feasible_window"]
    assert fw["earliest"] < fw["latest"]

# ─── Case 5: 兩邊衝突，無可行窗口 ───
def test_conflict_no_feasible_window():
    slots = [
        make_slot("s1", "店A", "09:00", duration_minutes=90),
        make_slot("s2", "店B", "09:30", duration_minutes=90),
        make_slot("s3", "店C", "10:00", duration_minutes=90),
        # s1 結束 10:30，s3 開始 10:00，根本來不及
    ]
    result = _conflict_check_travel(slots, "s2", CATALOG_FAR)
    assert result is not None
    assert result["feasible_window"] is None
