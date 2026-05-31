from agent import _extract_user_time_window, _requested_meal_slots


def test_extract_global_time_window():
    tr = _extract_user_time_window("今天拉麵行程 7:00~21:00 謝謝")
    assert tr.start == "07:00"
    assert tr.end == "21:00"


def test_extract_global_time_window_natural_language_periods():
    tr = _extract_user_time_window("早上7點到晚上9點，幫我排拉麵")
    assert tr.start == "07:00"
    assert tr.end == "21:00"


def test_extract_start_only_dots():
    """'7點開始' anchors morning without a range — meal expansion uses this for breakfast-first slots."""
    tr = _extract_user_time_window("京都 7點開始 安排五餐")
    assert tr.start == "07:00"
    assert tr.end is None


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


def test_requested_meal_slots_five_meals_morning_dots_start():
    slots = _requested_meal_slots("京都 7點開始 安排五餐")
    assert slots == ["breakfast", "lunch", "tea", "dinner", "late_night"]


def test_combine_itinerary_clock_maps_cross_midnight_roll_to_excursion_day():
    from datetime import datetime

    from dp_solver import _combine_itinerary_clock

    trip = datetime(2026, 5, 2, 7, 0)
    rolled = datetime(2026, 5, 3, 7, 15)
    aligned = _combine_itinerary_clock(trip, rolled)
    assert aligned == datetime(2026, 5, 2, 7, 15)

