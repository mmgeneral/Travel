"""Smoke test for the canonical Phase‑1 positive‑control harness.

This test uses the exact matched pairwise generator from
experiments.phase1_positive_control and the *real* refit_laplace
implementation.
"""
from __future__ import annotations

from experiments.phase1_positive_control import run_smoke


def test_smoke_runs_and_returns_expected_aggregate():
    raw, agg, pooled = run_smoke(worlds=3, N=40, checkpoints=(10, 30, 40))

    assert not raw.empty
    assert not agg.empty
    assert not pooled.empty

    n_counts = raw.groupby("n").size().to_dict()
    assert n_counts == {10: 3, 30: 3, 40: 3}

    required = {
        "n",
        "median_l2",
        "iqr_l2",
        "median_cos",
        "median_norm_ratio",
        "median_max_eig_sigma",
        "median_min_eig_negH",
        "median_grad_norm",
    }
    assert required.issubset(set(agg.columns))

    assert all(raw["max_eig_sigma"] <= 1.0 + 1e-6)
    assert all(pooled["std"] > 0.0)
