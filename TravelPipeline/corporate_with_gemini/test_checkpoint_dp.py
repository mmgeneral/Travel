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
    """For any i,j the probabilities of no‑error and first‑error sum to 1."""
    p_success = [0.9, 0.8, 0.7]
    L = len(p_success)
    for i in range(L):
        for j in range(i + 1, L + 1):
            inclusive_end = min(j, L - 1)

            prob_no_error = 1.0
            for k in range(i + 1, inclusive_end + 1):
                prob_no_error *= p_success[k]

            prob_first_err_sum = 0.0
            for m in range(i + 1, inclusive_end + 1):
                prob_until_m = 1.0
                for k in range(i + 1, m):
                    prob_until_m *= p_success[k]
                prob_first_err_sum += prob_until_m * (1.0 - p_success[m])

            assert abs(prob_no_error + prob_first_err_sum - 1.0) < 1e-9


def test_final_checkpoint_always_included() -> None:
    """Last slot must always appear in the returned checkpoint list."""
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
    """Empty list yields empty checkpoint list (no off‑by‑one crash)."""
    assert solve_checkpoints([], FixedBeliefStore([])) == []


def test_intermediate_checkpoints_appear_with_distance_scaled_diagnosis() -> None:
    """With realistic parameters, the DP should NOT collapse to 'always skip to the end'.

    Regression test for the bug where flat t_diagnose caused solve_checkpoints
    to always place a single checkpoint at the last slot regardless of risk.
    """
    slots = [
        Slot(slot_id=f"s{i}", day=1, period="x", risk_tags=[f"tag{i}"],
             affected_scope=[f"s{i}"], redo_cost_seconds=1.0)
        for i in range(5)
    ]
    store = FixedBeliefStore([0.7, 0.7, 0.9, 0.85, 0.85])
    result = solve_checkpoints(slots, store, t_confirm=1.0, t_diagnose_per_state=1.0)

    # Must have more than just the final mandatory checkpoint.
    assert len(result) > 1, (
        f"Expected multiple checkpoints (distance-scaled diagnosis cost should "
        f"favor intermediate checkpoints), got only {result}"
    )
