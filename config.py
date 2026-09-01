"""Central configuration constants for the research prototype.

Phase A/B/C constants live here so they are not scattered across modules.
"""

# ----------------------------------------------------------------------
# Phase A (frozen)
# ----------------------------------------------------------------------
# Phase 1: channel-specific lapse probabilities.
# Choice / implicit pairwise likelihood has no lapse (freeze).
# Prompted / critique / report channels keep their report-noise semantics.
LAMBDA_CHOICE = 0.0
LAMBDA_REPORT = 0.1
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
# Phase D3 — DV 量測
# ----------------------------------------------------------------------
# Dominant-block threshold δ：僅當 |Δu_T*| 與 |Δu_C*| 之絕對值差 > δ 時，
# 才報告 dominant-block accuracy。
DOMINANT_BLOCK_DELTA = 0.1

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

# ----------------------------------------------------------------------
# Phase D4 — sweep runner
# ----------------------------------------------------------------------
# Default number of independent repetitions per parameter combination.
SWEEP_REPEATS = 3

# Headline condition for main experiments: p_crit = 0 (implicit-only).
SWEEP_HEADLINE_P_CRIT = 0.0
SWEEP_HEADLINE_SIGMA_ITEM = 0.0

# c_int values to sweep, anchored on the current pre-calibration C_INT.
# The C4 anchoring step recommends grid fractions of the median top-2 gap;
# here we sweep both the baseline and its fractions.
SWEEP_C_INT_GRID = (
    C_INT,
    C_INT * 0.02,
    C_INT * 0.10,
    C_INT * 0.20,
)

# Experiment arms to iterate over in the sweep output.
SWEEP_ARMS = ("C0", "C1", "C2", "C3", "C4")
