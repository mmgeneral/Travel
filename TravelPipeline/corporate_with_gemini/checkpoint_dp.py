"""DP solver for Layer‑1 checkpoint scheduling (v1 prototype).

The recurrence is the classic backward‑induction structure of
"Expectation of remaining user time" but with the redo term based on
``affected_scope`` and bounded by the checkpoint position.

``[SIMPLIFIED]`` assumptions:
- t_confirm is a flat constant for every checkpoint.
- t_diagnose_per_state is the cost of inspecting a single state while
  walking backwards; the total diagnosis cost scales linearly with the
  distance to the first error.
- The DP recurses via ``dp[m+1]`` after the first error at slot ``m``.
- There is no sentinel ``j == n`` branch; every chosen ``j`` is a real
  slot index, and the final slot is naturally covered as part of some
  checkpoint span.
- The solver returns a list of slot_ids to pause at.  This list is
  deterministic for the given slots/belief store.
"""

from __future__ import annotations

from typing import Dict, List

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

    # dp[i] = minimal expected remaining user time given slots 0..i‑1 are
    #         already verified and slot i (if i < n) has not yet been examined.
    # dp[n] = 0 (base case: all n slots verified).
    dp = [0.0] * (n + 1)
    next_checkpoint = [None] * (n + 1)  # chosen real slot index for next checkpoint

    for i in range(n - 1, -1, -1):
        best = float("inf")
        best_j = None

        # choose a real slot j in [i, n‑1] as the next checkpoint
        for j in range(i, n):
            # probability all slots from i to j inclusive are correct
            prob_ok = 1.0
            for k in range(i, j + 1):
                prob_ok *= p_success[k]

            expected_cost = prob_ok * (t_confirm + dp[j + 1])

            prob_until_m = 1.0
            for m in range(i, j + 1):
                prob_fail_m = prob_until_m * (1.0 - p_success[m])
                diagnose_cost = t_diagnose_per_state * (m - i + 1)

                redo_cost = 0.0
                for scoped in slots[m].affected_scope:
                    idx = slot_index.get(scoped)
                    if idx is None or idx < m or idx > j:
                        continue
                    redo_cost += slots[idx].redo_cost_seconds

                expected_cost += prob_fail_m * (diagnose_cost + redo_cost + dp[m + 1])
                prob_until_m *= p_success[m]

            if expected_cost < best:
                best = expected_cost
                best_j = j

        dp[i] = best
        next_checkpoint[i] = best_j

    # reconstruct checkpoint locations (i advances to j+1)
    checkpoint_slot_ids: List[str] = []
    i = 0
    while i < n:
        j = next_checkpoint[i]
        if j is None:
            # Should never happen for a correct DP, but guard for safety.
            break
        checkpoint_slot_ids.append(slots[j].slot_id)
        i = j + 1

    return checkpoint_slot_ids
