"""Sanity checks for the checkpoint DP solver."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from checkpoint_dp import solve_checkpoints
from slot_model import Slot


class FixedBeliefStore:
    """Minimal belief store that returns probabilities in slot order."""

    def __init__(self, probabilities: list[float]):
        self.probabilities = probabilities
        self._index = 0

    def probability_for_tags(self, tags) -> float:
        prob = self.probabilities[self._index]
        self._index += 1
        return prob


def test_probability_sums_to_one_for_any_subrange() -> None:
    """For any i,j: P_ok(i,j) + sum of P_fail_at(i,m) for m in [i,j] == 1."""
    p_success = [0.9, 0.8, 0.7, 0.85]
    n = len(p_success)
    for i in range(n):
        for j in range(i, n):
            prob_ok = 1.0
            for k in range(i, j + 1):
                prob_ok *= p_success[k]
            prob_until_m = 1.0
            total_fail = 0.0
            for m in range(i, j + 1):
                total_fail += prob_until_m * (1.0 - p_success[m])
                prob_until_m *= p_success[m]
            assert abs(prob_ok + total_fail - 1.0) < 1e-9


def test_final_slot_always_covered() -> None:
    """The last real slot must always be part of some checkpoint's span."""
    slots = [
        Slot(slot_id="day1_lunch", day=1, period="lunch", risk_tags=["t1"]),
        Slot(slot_id="day1_dinner", day=1, period="dinner", risk_tags=["t2"]),
        Slot(slot_id="day2_breakfast", day=2, period="breakfast", risk_tags=["t3"]),
    ]
    store = FixedBeliefStore([0.9, 0.8, 0.7])
    result = solve_checkpoints(slots, store)
    assert result
    assert result[-1] == slots[-1].slot_id


def test_empty_slots_return_empty() -> None:
    assert solve_checkpoints([], FixedBeliefStore([])) == []


def test_intermediate_checkpoints_emerge_with_bounded_cascading_redo() -> None:
    """When redo cost genuinely grows with detection delay (affected_scope
    cascades forward, bounded by the checkpoint position), the DP should
    place more than just the final mandatory checkpoint.

    Regression test for the bug where redo cost wasn't bounded by the
    checkpoint position j, which removed the incentive for early detection
    and caused solve_checkpoints to always collapse to a single final
    checkpoint.
    """
    ids = [f"s{i}" for i in range(5)]
    slots = [
        Slot(slot_id=ids[i], day=1, period="x", risk_tags=[f"tag{i}"],
             affected_scope=ids[i:], redo_cost_seconds=1.0)
        for i in range(5)
    ]
    store = FixedBeliefStore([0.7, 0.7, 0.9, 0.85, 0.85])
    result = solve_checkpoints(slots, store, t_confirm=1.0, t_diagnose_per_state=1.0)
    assert len(result) > 1, (
        f"Expected multiple checkpoints when redo cost cascades and is "
        f"bounded by checkpoint position, got only {result}"
    )
