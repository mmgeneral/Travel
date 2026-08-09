"""Slot data model for checkpoint scheduling (v1 prototype).

Known risk_tags:
    Content risks (isolated):
        invalid_name, rating_threshold, budget_ceiling, dietary_filter
    Temporal risks (cascade forward):
        drifted_anchor, time_overflow, operating_hours_strict, reservation_required

    The former ``multi_constraint`` tag has been removed; use the specific
    content and/or temporal tags above instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

KNOWN_RISK_TAGS = {
    "invalid_name",
    "rating_threshold",
    "budget_ceiling",
    "dietary_filter",
    "drifted_anchor",
    "time_overflow",
    "operating_hours_strict",
    "reservation_required",
}

TEMPORAL_RISK_TAGS = {
    "drifted_anchor",
    "time_overflow",
    "operating_hours_strict",
    "reservation_required",
}


@dataclass
class Slot:
    """A unit of the travel itinerary that may need a user checkpoint.

    Attributes:
        slot_id: unique identifier, e.g. "day2_dinner".
        day: itinerary day (int).
        period: "breakfast", "lunch", "dinner", "activity", ...
        risk_tags: known/suspected failure modes (see module docs).
        affected_scope: slot_ids that would need re‑planning if this slot fails.
        redo_cost_seconds: estimated re‑planning time if this slot fails.
    """

    slot_id: str
    day: int
    period: str
    risk_tags: List[str] = field(default_factory=list)
    affected_scope: List[str] = field(default_factory=list)
    redo_cost_seconds: float = 0.0

    def __repr__(self) -> str:
        return (
            f"Slot(slot_id={self.slot_id!r}, day={self.day}, "
            f"period={self.period!r}, risk_tags={self.risk_tags!r})"
        )
