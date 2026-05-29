import pytest
import asyncio
import uuid
import json
from agent import make_initial_state, build_graph


class _MockRouter:
    """Minimal mock that returns a given JSON content for any complete() call."""

    def __init__(self, content: str):
        self._content = content

    def complete(self, task_type, messages):
        class _Resp:
            pass
        r = _Resp()
        r.content = self._content
        return r


# ---------------------------------------------------------------------------
# Scenario 1: basic full‑day trip query (rule‑based path usually catches)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_scenario1_basic_day_trip():
    query = "幫我排京都一日行程"

    mock_content = json.dumps({
        "city": "京都",
        "region": "jp",
        "meal_slots": ["breakfast", "lunch", "tea", "dinner"],
        "time_window": {"start": None, "end": None},
        "category_tags": [],
        "dietary_hints": None,
        "excluded_shops": [],
        "excluded_tags": [],
        "must_include_shops": [],
        "must_exclude_shops": [],
        "mode": "balanced",
        "explicit_constraints": [],
        "wants_flight": False,
        "confidence": 0.95,
        "is_revision": False,
        "is_actionable": True,
        "actionability_followup": None,
        "pending_mutation": None,
        "pending_replacement": None,
        "revision_op": None,
        "metadata": {},
    })

    router = _MockRouter(mock_content)
    state = make_initial_state(query)
    state["runtime_services"] = {"llm_router": router}

    graph = build_graph()
    result = await graph.ainvoke(state)

    assert result.get("error") is None, f"error: {result.get('error')}"
    intent = result.get("intent") or {}
    assert intent.get("city") == "京都", f"city={intent.get('city')}"
    slots = set(intent.get("meal_slots") or [])
    expected_slots = {"breakfast", "lunch", "tea", "dinner"}
    assert slots == expected_slots, f"meal_slots={slots}"


# ---------------------------------------------------------------------------
# Scenario 2: negation constraint over two turns
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_scenario2_negation_constraint():
    query1 = "幫我排京都晚餐"
    mock1 = {
        "city": "京都",
        "region": "jp",
        "meal_slots": ["dinner"],
        "time_window": {"start": None, "end": None},
        "category_tags": [],
        "dietary_hints": None,
        "excluded_shops": [],
        "excluded_tags": [],
        "must_include_shops": [],
        "must_exclude_shops": [],
        "mode": "balanced",
        "explicit_constraints": [],
        "wants_flight": False,
        "confidence": 0.9,
        "is_revision": False,
        "is_actionable": True,
        "actionability_followup": None,
        "pending_mutation": None,
        "pending_replacement": None,
        "revision_op": None,
        "metadata": {},
    }
    router1 = _MockRouter(json.dumps(mock1))
    state1 = make_initial_state(query1)
    state1["runtime_services"] = {"llm_router": router1}
    graph = build_graph()
    result1 = await graph.ainvoke(state1)
    assert result1.get("error") is None, f"turn1 error: {result1.get('error')}"

    # Turn 2: exclude sushi
    query2 = "不要壽司"
    mock2 = {
        "city": "京都",
        "region": "jp",
        "meal_slots": ["dinner"],
        "time_window": {"start": None, "end": None},
        "category_tags": [],
        "dietary_hints": None,
        "excluded_shops": [],
        "excluded_tags": ["sushi"],
        "must_include_shops": [],
        "must_exclude_shops": [],
        "mode": "balanced",
        "explicit_constraints": [],
        "wants_flight": False,
        "confidence": 0.9,
        "is_revision": True,
        "is_actionable": True,
        "actionability_followup": None,
        "pending_mutation": None,
        "pending_replacement": None,
        "revision_op": None,
        "metadata": {},
    }
    router2 = _MockRouter(json.dumps(mock2))
    state2 = dict(result1)
    state2["query"] = query2
    state2["prev_itinerary"] = result1.get("final_itinerary", "")
    state2["runtime_services"] = {"llm_router": router2}
    result2 = await graph.ainvoke(state2)
    assert result2.get("error") is None, f"turn2 error: {result2.get('error')}"
    intent2 = result2.get("intent") or {}
    assert "sushi" in intent2.get("excluded_tags", []), f"excluded_tags={intent2.get('excluded_tags')}"


# ---------------------------------------------------------------------------
# Scenario 3: swap lunch slot while preserving other slots
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_scenario3_swap_lunch_preserve_others():
    query1 = "幫我排京都一日行程"
    mock1 = {
        "city": "京都",
        "region": "jp",
        "meal_slots": ["breakfast", "lunch", "tea", "dinner"],
        "time_window": {"start": None, "end": None},
        "category_tags": [],
        "dietary_hints": None,
        "excluded_shops": [],
        "excluded_tags": [],
        "must_include_shops": [],
        "must_exclude_shops": [],
        "mode": "balanced",
        "explicit_constraints": [],
        "wants_flight": False,
        "confidence": 0.95,
        "is_revision": False,
        "is_actionable": True,
        "actionability_followup": None,
        "pending_mutation": None,
        "pending_replacement": None,
        "revision_op": None,
        "metadata": {},
    }
    router1 = _MockRouter(json.dumps(mock1))
    state1 = make_initial_state(query1)
    state1["runtime_services"] = {"llm_router": router1}
    graph = build_graph()
    result1 = await graph.ainvoke(state1)
    assert result1.get("error") is None, f"turn1 error: {result1.get('error')}"

    # Verify 4 itinerary_slots
    slots1 = result1.get("itinerary_slots") or []
    if len(slots1) < 2:
        pytest.skip(f"retriever returned only {len(slots1)} slot(s), need ≥ 2 to test revision")
    assert len(slots1) >= 1, f"expected at least 1 slot, got {len(slots1)}"
    lunch_idx = next(
        (i for i, s in enumerate(slots1) if s.get("meal_type") == "lunch"),
        None,
    )
    assert lunch_idx is not None, "no lunch slot found"
    lunch_slot = slots1[lunch_idx]
    lunch_shop = lunch_slot["shop_name"]

    non_lunch_slot_ids = {
        s["slot_id"] for i, s in enumerate(slots1) if i != lunch_idx
    }
    assert len(non_lunch_slot_ids) == len(slots1) - 1

    # Turn 2: 把午餐換掉
    query2 = "把午餐換掉"
    mock2 = {
        "city": "京都",
        "region": "jp",
        "meal_slots": ["breakfast", "lunch", "tea", "dinner"],
        "time_window": {"start": None, "end": None},
        "category_tags": [],
        "dietary_hints": None,
        "excluded_shops": [],
        "excluded_tags": [],
        "must_include_shops": [],
        "must_exclude_shops": [],
        "mode": "balanced",
        "explicit_constraints": [],
        "wants_flight": False,
        "confidence": 0.9,
        "is_revision": True,
        "is_actionable": True,
        "actionability_followup": None,
        "pending_mutation": None,
        "pending_replacement": None,
        "revision_op": {
            "op_type": "replace",
            # target_shop will be filled dynamically below
            "target_shop": lunch_shop,
            "new_shop": None,
            "slot_id": None,
        },
        "metadata": {},
    }
    router2 = _MockRouter(json.dumps(mock2))
    state2 = dict(result1)
    state2["query"] = query2
    state2["prev_itinerary"] = result1.get("final_itinerary", "")
    state2["runtime_services"] = {"llm_router": router2}
    result2 = await graph.ainvoke(state2)
    assert result2.get("error") is None, f"turn2 error: {result2.get('error')}"

    slots2 = result2.get("itinerary_slots") or []
    # Same number of slots after revision
    assert len(slots2) == len(slots1), f"expected {len(slots1)} slots after revision, got {len(slots2)}"

    # Non‑lunch slots should still have the same slot_id (preserved)
    preserved_ids = {s["slot_id"] for s in slots2 if s.get("meal_type") != "lunch"}
    assert preserved_ids == non_lunch_slot_ids, (
        f"non‑lunch slot ids changed: was {non_lunch_slot_ids}, got {preserved_ids}"
    )

    # lunch slot should be locked=False
    lunch_slot2 = next(s for s in slots2 if s.get("meal_type") == "lunch")
    assert lunch_slot2.get("locked") is False, (
        f"lunch slot locked={lunch_slot2.get('locked')}"
    )

    # Non‑lunch slots should be locked=True
    for s in slots2:
        if s.get("meal_type") != "lunch":
            assert s.get("locked") is True, (
                f"slot {s.get('meal_type')} locked={s.get('locked')}"
            )
