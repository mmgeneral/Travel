import math
import pytest
from datetime import datetime

from dp_solver import _haversine_km, _combine_itinerary_clock


# ── _haversine_km ────────────────────────────────────────

def test_haversine_same_point_is_zero():
    assert _haversine_km(35.0, 135.0, 35.0, 135.0) == pytest.approx(0.0, abs=1e-6)


def test_haversine_known_distance():
    # 台北（25.04, 121.51）到 東京（35.68, 139.69）約 2100 km
    dist = _haversine_km(25.04, 121.51, 35.68, 139.69)
    assert 2000 < dist < 2300


def test_haversine_symmetry():
    d1 = _haversine_km(35.0, 135.0, 36.0, 136.0)
    d2 = _haversine_km(36.0, 136.0, 35.0, 135.0)
    assert d1 == pytest.approx(d2, rel=1e-6)


def test_haversine_small_distance():
    # 約 1 度緯度差 ≈ 111 km
    dist = _haversine_km(35.0, 135.0, 36.0, 135.0)
    assert 100 < dist < 120


# ── _combine_itinerary_clock ─────────────────────────────

def test_combine_clock_same_day():
    trip = datetime(2026, 6, 1, 9, 0)
    t = datetime(2026, 6, 1, 12, 30)
    result = _combine_itinerary_clock(trip, t)
    assert result == datetime(2026, 6, 1, 12, 30)


def test_combine_clock_rolls_over_midnight():
    # t 是隔天但時間早於 trip_start，應該對齊到 trip 的當天
    trip = datetime(2026, 6, 1, 7, 0)
    rolled = datetime(2026, 6, 2, 7, 15)
    result = _combine_itinerary_clock(trip, rolled)
    assert result == datetime(2026, 6, 1, 7, 15)


def test_combine_clock_preserves_time_component():
    trip = datetime(2026, 6, 5, 8, 0)
    t = datetime(2026, 6, 5, 19, 45)
    result = _combine_itinerary_clock(trip, t)
    assert result.hour == 19
    assert result.minute == 45
