import pytest
from unittest.mock import patch
from intent_parser import Intent, RevisionOp

from agent import make_initial_state, node_route_intent


# ---------------------------------------------------------------------------
# helpers – provided by specification
# ---------------------------------------------------------------------------
def make_state_with_slots(slots, intent_override=None):
    """Build a minimal AgentState with given itinerary_slots and intent."""
    state = make_initial_state("test")
    state["itinerary_slots"] = slots
    state["intent"] = {
        "city": "京都",
        "region": "jp",
        "meal_slots": [],
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
        "revision_op": None,
        "confirm_op": None,
        "metadata": {},
    }
    if intent_override:
        state["intent"].update(intent_override)
    state["runtime_services"] = {"llm_router": None}
    return state


def make_slot(slot_id, shop_name, meal_type="lunch",
              user_locked=False, session_locked=False):
    return {
        "slot_id": slot_id,
        "shop_name": shop_name,
        "meal_type": meal_type,
        "user_locked": user_locked,
        "session_locked": session_locked,
        "start_time": None,
        "duration_minutes": 90,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_must_include_sets_user_locked():
    """Case 1: must_include_shops → user_locked=True"""
    slots = [
        make_slot("s1", "ShopA", meal_type="lunch"),
        make_slot("s2", "ShopB", meal_type="dinner"),
    ]
    state = make_state_with_slots(slots)

    mock_intent = Intent(
        city="京都",
        region="jp",
        meal_slots=[],
        time_window={"start": None, "end": None},
        category_tags=[],
        dietary_hints=None,
        excluded_shops=[],
        excluded_tags=[],
        must_include_shops=["ShopA"],
        must_exclude_shops=[],
        mode="balanced",
        explicit_constraints=[],
        wants_flight=False,
        confidence=0.9,
        is_revision=True,
        is_actionable=True,
        actionability_followup=None,
        revision_op=None,
        confirm_op=None,
        metadata={},
    )

    with patch("agent.parse_intent", return_value=mock_intent):
        result = node_route_intent(state)

    slots_out = result["itinerary_slots"]
    shopA = next(s for s in slots_out if s["shop_name"] == "ShopA")
    shopB = next(s for s in slots_out if s["shop_name"] == "ShopB")
    assert shopA["user_locked"] is True
    assert shopB["user_locked"] is False


def test_must_exclude_unlocks():
    """Case 2: must_exclude_shops → user_locked cleared"""
    slots = [
        make_slot("s1", "ShopA", meal_type="lunch", user_locked=True),
        make_slot("s2", "ShopB", meal_type="dinner", user_locked=True),
    ]
    state = make_state_with_slots(slots)

    mock_intent = Intent(
        city="京都",
        region="jp",
        meal_slots=[],
        time_window={"start": None, "end": None},
        category_tags=[],
        dietary_hints=None,
        excluded_shops=[],
        excluded_tags=[],
        must_include_shops=[],
        must_exclude_shops=["ShopA"],
        mode="balanced",
        explicit_constraints=[],
        wants_flight=False,
        confidence=0.9,
        is_revision=True,
        is_actionable=True,
        actionability_followup=None,
        revision_op=None,
        confirm_op=None,
        metadata={},
    )

    with patch("agent.parse_intent", return_value=mock_intent):
        result = node_route_intent(state)

    slots_out = result["itinerary_slots"]
    shopA = next(s for s in slots_out if s["shop_name"] == "ShopA")
    shopB = next(s for s in slots_out if s["shop_name"] == "ShopB")
    # ShopA is excluded → should be unlocked
    assert shopA["user_locked"] is False
    # ShopB stays locked as before (intact)
    assert shopB["user_locked"] is True


def test_session_locked_cleared():
    """Case 3: revision_op with replace … clear session_locked for all slots"""
    slots = [
        make_slot("s1", "ShopA", meal_type="lunch", session_locked=True),
        make_slot("s2", "ShopB", meal_type="dinner", session_locked=True),
        make_slot("s3", "ShopC", meal_type="lunch", session_locked=True),
    ]
    state = make_state_with_slots(slots)

    rev = RevisionOp(
        op_type="replace",
        target_shop="ShopC",
        new_shop="ShopD",
        slot_id=None,
    )
    mock_intent = Intent(
        city="京都",
        region="jp",
        meal_slots=[],
        time_window={"start": None, "end": None},
        category_tags=[],
        dietary_hints=None,
        excluded_shops=[],
        excluded_tags=[],
        must_include_shops=[],
        must_exclude_shops=[],
        mode="balanced",
        explicit_constraints=[],
        wants_flight=False,
        confidence=0.9,
        is_revision=True,
        is_actionable=True,
        actionability_followup=None,
        revision_op=rev,
        confirm_op=None,
        metadata={},
    )

    with patch("agent.parse_intent", return_value=mock_intent):
        result = node_route_intent(state)

    slots_out = result["itinerary_slots"]
    for s in slots_out:
        assert s["session_locked"] is False, f"{s['shop_name']} still session_locked"


def test_user_locked_persists():
    """Case 4: revision_op does not clear user_locked"""
    slots = [
        make_slot("s1", "ShopA", meal_type="lunch", user_locked=True),
        make_slot("s2", "ShopB", meal_type="dinner", user_locked=False),
    ]
    state = make_state_with_slots(slots)

    rev = RevisionOp(
        op_type="replace",
        target_shop="ShopB",
        new_shop="ShopC",
        slot_id=None,
    )
    mock_intent = Intent(
        city="京都",
        region="jp",
        meal_slots=[],
        time_window={"start": None, "end": None},
        category_tags=[],
        dietary_hints=None,
        excluded_shops=[],
        excluded_tags=[],
        must_include_shops=[],
        must_exclude_shops=[],
        mode="balanced",
        explicit_constraints=[],
        wants_flight=False,
        confidence=0.9,
        is_revision=True,
        is_actionable=True,
        actionability_followup=None,
        revision_op=rev,
        confirm_op=None,
        metadata={},
    )

    with patch("agent.parse_intent", return_value=mock_intent):
        result = node_route_intent(state)

    slots_out = result["itinerary_slots"]
    shopA = next(s for s in slots_out if s["shop_name"] == "ShopA")
    shopB = next(s for s in slots_out if s["shop_name"] == "ShopB")
    # user_locked should remain as it was originally
    assert shopA["user_locked"] is True
    assert shopB["user_locked"] is False
