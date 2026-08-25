"""Canonical feature registry for the frozen Phase A1–A4 model.

No external dependencies; used by decision_engine.py (A1),
evidence.py (A2), and likelihood.py (A3/A4).
"""

FEATURE_NAMES = [
    "cuisine_match",
    "fame_touristy",
    "heaviness",
    "travel_min",
    "price_level",
    "queue_wait",
]

FEATURE_NAME_TO_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}

TASTE_INDICES = [0, 1, 2]
CONTEXT_INDICES = [3, 4, 5]
