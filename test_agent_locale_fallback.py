"""Locale / city fallback and flight-vs-food routing behavior."""

from __future__ import annotations

import asyncio
import json
import os

import pytest

from agent import (
    _extract_city_from_query,
    _is_flight_booking_intent,
    build_graph,
    make_initial_state,
)


def test_explicit_city_beats_user_locale_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEFAULT_LOCALE_CITY", "Taipei")
    city, region = _extract_city_from_query("京都 咖啡下午茶", user_locale="Taipei")
    assert city == "京都"
    assert region == "jp"


def test_user_locale_beats_default_locale_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEFAULT_LOCALE_CITY", "Taipei")
    city, region = _extract_city_from_query("好吃拉麵推薦", user_locale="Kyoto")
    assert city == "京都"
    assert region == "jp"


def test_default_locale_env_before_kyoto_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEFAULT_LOCALE_CITY", "Taipei")
    city, region = _extract_city_from_query("匿名美食探索", user_locale=None)
    assert city == "台北"
    assert region == "tw"


def test_kyoto_when_no_locale_signals(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEFAULT_LOCALE_CITY", raising=False)
    city, region = _extract_city_from_query("匿名美食探索", user_locale=None)
    assert city == "京都"
    assert region == "jp"


def test_make_initial_state_carries_user_locale() -> None:
    st = make_initial_state("query", user_locale="Osaka")
    assert st["user_locale"] == "Osaka"


def test_pure_food_query_skips_duffel_audit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Food-only routing must not append Duffel-related transit_audit lines."""
    monkeypatch.setenv("OFFLINE_MODE", "1")
    monkeypatch.delenv("DEFAULT_LOCALE_CITY", raising=False)
    graph = build_graph()
    initial = make_initial_state("京都駅附近午餐 拉麵")
    result = asyncio.run(graph.ainvoke(initial))
    audit_text = json.dumps(result.get("transit_audit") or [])
    assert "duffel" not in audit_text.lower()
    assert _is_flight_booking_intent("京都駅附近午餐 拉麵") is False


def test_flight_keyword_triggers_flight_intent() -> None:
    assert _is_flight_booking_intent("Book TPE to SFO via NRT") is True
    assert _is_flight_booking_intent("機票 台北東京") is True
    assert _is_flight_booking_intent("cheap flight to Osaka") is True
