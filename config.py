"""Central configuration constants for the research prototype.

Phase A/B/C constants live here so they are not scattered across modules.
"""

# ----------------------------------------------------------------------
# Phase A (frozen)
# ----------------------------------------------------------------------
LAMBDA = 0.1
TAU = 1.0
KAPPA = 0.0

# ----------------------------------------------------------------------
# Phase B
# ----------------------------------------------------------------------
M = 5  # top-M candidates after posterior rerank

# ----------------------------------------------------------------------
# Phase C
# ----------------------------------------------------------------------
C1_ETA = 0.25          # active-dim threshold (z-units)
C_INT = 0.05           # current pre-calibration interruption cost
MC_DRAWS = 200         # number of posterior draws for EVOI expectation
EVOI_DIAGNOSTIC_THRESHOLD = 20  # after this many samples, print diagnostics

# ----------------------------------------------------------------------
# Phase C4 — c_int anchoring
# ----------------------------------------------------------------------
# Suggested grid fractions relative to the median top‑2 S_B gap.
C_INT_GRID_FRACTIONS = (0.02, 0.10, 0.20)

# ----------------------------------------------------------------------
# Phase D — synthetic user generator
# ----------------------------------------------------------------------
# σ²_item grid (variance) for item residual sensitivity tests.
SIGMA_ITEM_GRID = (0.0, 0.25, 1.0, 4.0)

# critique emission probabilities for spontaneous critique frequency.
P_CRIT_GRID = (0.0, 0.3, 0.6)

# ----------------------------------------------------------------------
# Phase D2 — experiment arms & gate switch
# ----------------------------------------------------------------------
# Current arm controlling the ask decision:
#   C0 = implicit-only  (gate always continue)
#   C1 = always-ask-on-eligible
#   C2 = EVOI-gated (current production behavior)
#   C3 = explicit pairwise baseline (EVOI-gated, same ask rule as C2)
#   C4 = no-learning control (EVOI-gated, same ask rule as C2)
EXPERIMENT_ARM = "C2"

# When True, evaluate_gate always returns ASK (if eligible). Used for the
# regression test that proves C1/C2 trajectories are bit-identical when
# the ask decision is forced.
FORCE_ASK = False
