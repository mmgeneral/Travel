from agent import _extract_user_time_window, _requested_meal_slots


def test_extract_global_time_window():
    tr = _extract_user_time_window("今天拉麵行程 7:00~21:00 謝謝")
    assert tr.start == "07:00"
    assert tr.end == "21:00"


def test_extract_global_time_window_natural_language_periods():
    tr = _extract_user_time_window("早上7點到晚上9點，幫我排拉麵")
    assert tr.start == "07:00"
    assert tr.end == "21:00"


def test_requested_meal_slots_from_count_without_explicit_slots():
    slots = _requested_meal_slots("幫我排三餐就好")
    assert slots == ["lunch", "tea", "dinner"]


def test_requested_meal_slots_three_ramen_with_morning_hint():
    slots = _requested_meal_slots("三餐拉麵，早上7點開始")
    assert slots == ["breakfast", "lunch", "dinner"]


def test_requested_meal_slots_forced_breakfast_from_time_window():
    slots = _requested_meal_slots("7:00~21:00 拉麵行程")
    assert slots
    assert slots[0] == "breakfast"
