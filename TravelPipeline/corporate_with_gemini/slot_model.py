"""Slot data model for checkpoint scheduling (v1 prototype)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


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
