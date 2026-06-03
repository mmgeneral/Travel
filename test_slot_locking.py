import pytest

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
        make_slot("slot1", "ShopA", "lunch", user_locked=False, session_locked=False),
        make_slot("slot2", "ShopB", "dinner", user_locked=False, session_locked=False),
    ]
    intent_override = {
        "must_include_shops": ["ShopA"],
        "is_revision": True,
    }
    state = make_state_with_slots(slots, intent_override)
    state = node_route_intent(state)

    # ShopA should be locked by the machine
    assert state["itinerary_slots"][0]["user_locked"] is True
    assert state["itinerary_slots"][1]["user_locked"] is False
