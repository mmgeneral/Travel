"""Tests for affected_scope.compute_affected_scope."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from slot_model import Slot
from affected_scope import compute_affected_scope


def test_content_risk_is_isolated():
    """A slot with only content-risk tags has affected_scope = [itself]."""
    slot = Slot(slot_id="s2", day=1, period="lunch", risk_tags=["invalid_name"])
    day_schedule = [
        {"slot_id": "s1", "locked": False},
        {"slot_id": "s2", "locked": False},
        {"slot_id": "s3", "locked": False},
    ]
    assert compute_affected_scope(slot, day_schedule) == ["s2"]


def test_temporal_risk_cascades_to_end_of_day():
    """A slot with a temporal-risk tag cascades to all subsequent slots
    when nothing is locked."""
    slot = Slot(slot_id="s2", day=1, period="lunch", risk_tags=["drifted_anchor"])
    day_schedule = [
        {"slot_id": "s1", "locked": False},
        {"slot_id": "s2", "locked": False},
        {"slot_id": "s3", "locked": False},
        {"slot_id": "s4", "locked": False},
    ]
    assert compute_affected_scope(slot, day_schedule) == ["s3", "s4"]


def test_temporal_risk_stops_before_locked_slot():
    """Cascade stops before (does not include) the next locked slot."""
    slot = Slot(slot_id="s1", day=1, period="lunch", risk_tags=["time_overflow"])
    day_schedule = [
        {"slot_id": "s1", "locked": False},
        {"slot_id": "s2", "locked": False},
        {"slot_id": "s3", "locked": True},
        {"slot_id": "s4", "locked": False},
    ]
    assert compute_affected_scope(slot, day_schedule) == ["s2"]


def test_mixed_content_and_temporal_tags_treated_as_temporal():
    """If a slot has both a content and a temporal tag, it cascades."""
    slot = Slot(slot_id="s1", day=1, period="lunch",
                risk_tags=["rating_threshold", "reservation_required"])
    day_schedule = [
        {"slot_id": "s1", "locked": False},
        {"slot_id": "s2", "locked": False},
    ]
    assert compute_affected_scope(slot, day_schedule) == ["s2"]


def test_slot_not_found_falls_back_to_self():
    slot = Slot(slot_id="ghost", day=1, period="lunch", risk_tags=["drifted_anchor"])
    day_schedule = [{"slot_id": "s1", "locked": False}]
    assert compute_affected_scope(slot, day_schedule) == ["ghost"]
