"""Tests for Google Places–driven opening time inference in agent.py."""

from __future__ import annotations

from agent import (
    _build_dynamic_shop_profile,
    _infer_close_time,
    _infer_open_time,
    _is_dynamic_time_unknown,
    _resolve_dynamic_open_time,
)


def test_infer_open_close_from_range_zh():
    s = "星期一: 11:00 – 21:00"
    assert _infer_open_time(s) == "11:00"
    assert _infer_close_time(s) == "21:00"


def test_infer_open_defaults_when_empty():
    assert _infer_open_time("") == "11:00"
    assert _infer_close_time("") == "22:00"


def test_resolve_dynamic_open_time_explicit_overrides_inference():
    place = {"open_time": "10:30", "opening_hours_today": "11:00 – 20:00"}
    assert _resolve_dynamic_open_time(place, "11:00 – 20:00") == "10:30"


def test_time_unknown_when_source_has_no_open_info():
    place = {"name": "Mystery", "opening_hours_today": ""}
    assert _is_dynamic_time_unknown(place, "")
    profile = _build_dynamic_shop_profile(place, region="jp")
    assert "TIME_UNKNOWN" in profile.tags
