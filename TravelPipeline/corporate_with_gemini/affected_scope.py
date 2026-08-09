"""Compute affected_scope for Slot objects based on risk tags (v1 prototype)."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from AnchorResolver import _build_schedule_by_day, _get_day_schedule  # type: ignore
from slot_model import Slot, TEMPORAL_RISK_TAGS

logger = logging.getLogger(__name__)


def compute_affected_scope(slot: Slot, day_schedule: List[dict]) -> List[str]:
    """Compute which slot_ids would need re-planning if `slot` fails.

    Content risks (no tag in TEMPORAL_RISK_TAGS): isolated, affected_scope = [slot itself].

    Temporal risks (at least one tag in TEMPORAL_RISK_TAGS): cascades forward
    through the same day's schedule, starting from the slot immediately after
    `slot`, stopping BEFORE (not including) the next slot with locked=True,
    or at the end of the day if no locked slot follows.

    Args:
        slot: the Slot being evaluated.
        day_schedule: the list of schedule-item dicts for slot's day, in
            itinerary order — same shape as AnchorResolver.py's
            `_get_day_schedule` / `_schedule_for_day` output. Each item is a
            dict with at least "slot_id" and "locked" keys.

    Returns:
        List of slot_ids in the affected scope (does not include `slot`
        itself for the temporal-risk case; content-risk case returns
        exactly [slot.slot_id]).
    """
    if not any(tag in TEMPORAL_RISK_TAGS for tag in slot.risk_tags):
        return [slot.slot_id]

    # Locate the current slot within the day schedule.
    index = None
    for i, item in enumerate(day_schedule):
        if item.get("slot_id") == slot.slot_id:
            index = i
            break

    if index is None:
        logger.warning(
            "Slot %s not found in day_schedule; falling back to isolated scope.",
            slot.slot_id,
        )
        return [slot.slot_id]

    affected: List[str] = []
    for i in range(index + 1, len(day_schedule)):
        item = day_schedule[i]
        if item.get("locked") is True:
            break
        slot_id = item.get("slot_id")
        if slot_id is not None:
            affected.append(slot_id)
    return affected


def compute_affected_scope_from_itinerary(
    slot: Slot,
    itinerary_json: Dict,
) -> List[str]:
    """Convenience wrapper that builds the day schedule from an itinerary JSON.

    Reuses AnchorResolver's `_build_schedule_by_day` and `_get_day_schedule`
    to obtain the day schedule for the slot's day.
    """
    schedule_by_day = _build_schedule_by_day(itinerary_json)
    day_schedule = _get_day_schedule(schedule_by_day, slot.day)
    return compute_affected_scope(slot, day_schedule)
