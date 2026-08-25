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
C_INT = 0.25           # interruption cost (EVOI subtracts this)
MC_DRAWS = 200         # number of posterior draws for EVOI expectation
EVOI_DIAGNOSTIC_THRESHOLD = 20  # after this many samples, print diagnostics

# ----------------------------------------------------------------------
# Phase C4 — c_int anchoring
# ----------------------------------------------------------------------
# Legacy hard-coded threshold that this diagnostic replaces (used only for
# reporting; the live system uses C_INT above).
C_INT_CURRENT = 0.05

# Suggested grid fractions relative to the median top‑2 S_B gap.
C_INT_GRID_FRACTIONS = (0.02, 0.10, 0.20)
