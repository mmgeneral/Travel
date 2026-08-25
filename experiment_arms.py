"""Phase D.2 — experiment arms metadata.

This module only declares the five arm names and their ask-mode mapping.
The actual gate decision lives in ``decision_engine.evaluate_gate`` and is
selected via ``config.EXPERIMENT_ARM``.
"""

ARM_NAMES = ("C0", "C1", "C2", "C3", "C4")

ARM_ASK_MODES = {
    "C0": "implicit_only",
    "C1": "always_ask_on_eligible",
    "C2": "evoi_gated",
    "C3": "explicit_pairwise_baseline",
    "C4": "no_learning_control",
}
