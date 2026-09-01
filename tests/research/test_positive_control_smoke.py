"""Smoke test for the exact matched pairwise positive-control experiment."""
from __future__ import annotations

from experiments.matched_pairwise_positive_control import run_smoke


def test_smoke_runs_and_returns_tables():
    raw, agg, pooled = run_smoke(worlds=3, N=40, checkpoints=(10, 30, 40))
    assert not raw.empty
    assert not agg.empty
    assert set(agg.columns) == {
        "n",
        "median_l2",
        "iqr_l2",
        "median_cos",
        "median_norm_ratio",
        "median_max_eig_sigma",
        "median_min_eig_negH",
        "median_grad_norm",
    }
    assert raw["n"].tolist() == [10] * 3 + [30] * 3 + [40] * 3
    assert all(raw["l2_error"] >= 0)
    assert all(raw["max_eig_sigma"] <= 1.0 + 1e-6)
    assert all(pooled["std"] > 0)
"""Smoke test for the canonical Phase‑1 positive‑control harness.

This test uses the exact matched pairwise generator from
experiments.phase1_positive_control and the *real* refit_laplace
implementation.  It does **not** depend on any stale lambda constants.
"""
from __future__ import annotations

import numpy as np
import pytest

from experiments.phase1_positive_control import run_smoke


def test_smoke_runs_and_returns_expected_aggregate():
    raw, agg, pooled = run_smoke(worlds=3, N=40, checkpoints=(10, 30, 40))

    assert not raw.empty
    assert not agg.empty
    assert not pooled.empty

    expected_n = [10] * 3 + [30] * 3 + [40] * 3
    assert raw["n"].tolist() == expected_n

    # Required aggregate columns per the Phase‑1 spec.
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

    # Choice‑only logs must satisfy Sigma <= I (up to numerical tolerance).
    assert all(raw["max_eig_sigma"] <= 1.0 + 1e-6)

    # Random‑projection calibration diagnostic should produce non‑zero spread.
    assert all(pooled["std"] > 0.0)
