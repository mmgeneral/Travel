"""DP solver for Layer‑1 checkpoint scheduling (v1 prototype).

The recurrence is the classic backward‑induction structure of
"Expectation of remaining user time" but with the redo term based on
``affected_scope`` rather than a linear position range.

``[SIMPLIFIED]`` assumptions:
- t_confirm and t_diagnose are flat constants (same for all periods).
- The DP treats a first‑error as a one‑time cost followed by continuing
  from the same interval start; a more faithful implementation would
  recursively call the DP from the error location.
- The solver returns a list of slot_ids to pause at.  This list is
  deterministic for the given slots/belief store.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from belief_store import BeliefStore
from slot_model import Slot


def solve_checkpoints(
    slots: List[Slot],
    belief_store: BeliefStore,
    t_confirm: float = 60.0,
    t_diagnose_per_state: float = 30.0,
) -> List[str]:
    """Return slot_ids where the UI should show a Layer‑1 confirmation dialog.

    Parameters
    ----------
    slots:
        Slots sorted in itinerary order (index 0 = first slot).
    belief_store:
        Feature‑indexed belief store with probabilities from ``probability_for_tags``.
    t_confirm:
        Flat per‑checkpoint confirmation cost (seconds).
    t_diagnose_per_state:
        Per‑state inspection cost used when walking backwards to locate the
        first error.  In the reference model this cost scales linearly with
        the distance between the last verified slot and the error slot.

    Returns
    -------
    list[str]
        Sorted slice of ``slot_id`` values where the pipeline should pause.
    """
    n = len(slots)
    if n == 0:
        return []

    slot_index: Dict[str, int] = {s.slot_id: i for i, s in enumerate(slots)}

    # per‑slot success probability using §3 feature‑indexed beliefs
    p_success = [belief_store.probability_for_tags(s.risk_tags) for s in slots]

    # dp[i] = minimal expected remaining user time after being verified at slot i
    #         (i ranges 0..n, where n is the virtual sentinel after the last slot)
    dp = [0.0] * (n + 1)
    next_checkpoint = [None] * (n + 1)

    for i in range(n - 1, -1, -1):
        best = float("inf")
        best_j = None

        # try every possible position j (i+1 … n) for the *next verified slot*
        # (j == n means we skip any further checkpoint until the end)
        for j in range(i + 1, n + 1):
            # probability that no error occurs in (i, j]
            # Includes success of slot j when j is a real checkpoint slot.
            inclusive_end = min(j, n - 1)
            prob_no_error = 1.0
            for k in range(i + 1, inclusive_end + 1):
                prob_no_error *= p_success[k]

            # expected confirm cost at j (only if we actually place a checkpoint at j)
            confirm_cost = t_confirm if j < n else 0.0
            expected_cost = prob_no_error * (confirm_cost + dp[j])

            # accumulate expected cost due to a first error at each m ∈ (i, inclusive_end]
            for m in range(i + 1, inclusive_end + 1):
                # probability that the first error happens at m
                prob_until_m = 1.0
                for k in range(i + 1, m):
                    prob_until_m *= p_success[k]
                prob_first_err = prob_until_m * (1.0 - p_success[m])

                # redo cost for all slots in affected_scope(m) that are at or after m
                redo_cost = 0.0
                for scoped in slots[m].affected_scope:
                    idx = slot_index.get(scoped)
                    if idx is None or idx < m:
                        # affected slot missing or lies before the failure => ignore for v1
                        continue
                    redo_cost += slots[idx].redo_cost_seconds

                diagnose_cost = t_diagnose_per_state * (m - i)
                expected_cost += prob_first_err * (diagnose_cost + redo_cost)

            if expected_cost < best:
                best = expected_cost
                best_j = j

        dp[i] = best
        next_checkpoint[i] = best_j

    # reconstruct checkpoint locations
    checkpoint_slot_ids: List[str] = []
    i = 0
    while i < n:
        j = next_checkpoint[i]
        if j is None:
            break
        if j == n:
            break
        checkpoint_slot_ids.append(slots[j].slot_id)
        i = j

    if n > 0 and (not checkpoint_slot_ids or checkpoint_slot_ids[-1] != slots[n - 1].slot_id):
        checkpoint_slot_ids.append(slots[n - 1].slot_id)

    return checkpoint_slot_ids
