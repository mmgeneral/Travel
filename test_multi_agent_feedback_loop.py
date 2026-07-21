"""Researcher ↔ critic loop anchors: candidates must span requested meal slots."""

from __future__ import annotations

import pytest

from decision_engine import ItinerarySynthesizer

from agent import (
    _build_shop_catalog,
    _call_researcher_prompt,
    _researcher_slot_anchor_names,
    _researcher_shop_eligible_any_tier,
)


def test_kyoto_seed_anchor_one_shop_per_standard_slot(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    seed = list(_build_shop_catalog())
    query = ""
    slots = ItinerarySynthesizer._normalize_slot_sequence(
        ["breakfast", "lunch", "tea", "dinner", "late_night"]
    )
    anchors = _researcher_slot_anchor_names(slots, seed, query=query)
    assert len(anchors) == len(slots)
    by_name = {s.name: s for s in seed}
    for slot, nm in zip(slots, anchors):
        assert _researcher_shop_eligible_any_tier(by_name[nm], slot, query=query)


async def test_researcher_fallback_respects_slot_count(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    seed = list(_build_shop_catalog())
    slots = ItinerarySynthesizer._normalize_slot_sequence(
        ["breakfast", "lunch", "tea", "dinner", "late_night"]
    )
    names, notes = await _call_researcher_prompt(
        "",
        seed,
        meal_slots=slots,
        seed_shops=seed,
        auditor_feedback="",
        iteration=1,
    )
    anchors = _researcher_slot_anchor_names(slots, seed, query="")

    assert len(names) >= len(slots)
    assert names[: len(anchors)] == anchors
    assert "slot_anchors=" in notes
    assert "heuristic_fallback_researcher" in notes

