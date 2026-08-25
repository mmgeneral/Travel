"""Phase D.4 sweep runner smoke test."""
import numpy as np
import pytest

import config
from phase_d_sweep import run_sweep


def test_sweep_returns_payload():
    res = run_sweep(
        p_crit_values=[0.0],
        sigma_item_values=[0.0],
        c_int_values=[0.05],
        repeats=3,
    )
    assert "raw_rows" in res
    assert "aggregates" in res
    assert "calibration" in res
    assert len(res["raw_rows"]) >= 3 * len(config.SWEEP_ARMS)
    assert "C0" in res["aggregates"]

    # each per-repetition raw row must contain the requested fields
    for row in res["raw_rows"]:
        for key in {
            "rep", "seed", "p_crit", "residual_multiplier", "realized_sigma2_item",
            "c_int", "arm", "clarification_count", "pairwise_count",
            "total_explicit_burden", "taste_recovery_error", "context_recovery_error",
            "cross_block_cov_norm", "cross_block_cov_fraction",
            "taste_contribution_error", "context_contribution_error",
            "dominant_block_correct", "dominant_block_eligible_count",
            "mean_regret", "cumulative_regret", "revision_count",
            "final_mu", "final_sigma_diag",
        }:
            assert key in row, f"Missing key {key} in row {row}"

    # C4 must remain at prior
    c4_rows = [r for r in res["raw_rows"] if r["arm"] == "C4"]
    for r in c4_rows:
        assert np.allclose(np.asarray(r["final_mu"]), np.zeros(6))
        assert np.allclose(np.asarray(r["final_sigma_diag"]), np.ones(6))

    # C3 must have non-zero pairwise query in at least some repetitions
    c3_rows = [r for r in res["raw_rows"] if r["arm"] == "C3"]
    assert any(r["pairwise_count"] > 0 for r in c3_rows)
