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
    aggs = res["aggregates"]
    assert any(k[3] == "C0" for k in aggs.keys())

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


def test_sweep_arm_order_invariance(monkeypatch):
    """Reverse SWEEP_ARMS and ensure per‑arm results are unchanged."""
    original_arms = list(config.SWEEP_ARMS)
    reversed_arms = list(reversed(original_arms))

    monkeypatch.setattr(config, "SWEEP_ARMS", original_arms)
    res_original = run_sweep(
        p_crit_values=[0.0],
        sigma_item_values=[0.0],
        c_int_values=[0.05],
        repeats=1,
    )

    monkeypatch.setattr(config, "SWEEP_ARMS", reversed_arms)
    res_reversed = run_sweep(
        p_crit_values=[0.0],
        sigma_item_values=[0.0],
        c_int_values=[0.05],
        repeats=1,
    )

    raw_orig = {r["arm"]: r for r in res_original["raw_rows"] if r["rep"] == 0}
    raw_rev = {r["arm"]: r for r in res_reversed["raw_rows"] if r["rep"] == 0}

    assert set(raw_orig.keys()) == set(raw_rev.keys())
    for arm in raw_orig:
        assert raw_orig[arm] == raw_rev[arm]


def test_calibration_includes_percentile():
    res = run_sweep(
        p_crit_values=[0.0],
        sigma_item_values=[0.0],
        c_int_values=[0.05],
        repeats=2,
    )
    cal = res["calibration"]
    assert "median_gap" in cal
    assert "q1_gap" in cal
    assert "q3_gap" in cal
    assert "iqr_gap" in cal
    assert "c_int_percentile" in cal
    assert "suggested_grid" in cal
    assert len(cal["suggested_grid"]) == 3
