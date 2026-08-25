"""Phase D.3 metrics unit tests."""
import numpy as np
import pytest

from phase_d_metrics import (
    contribution_errors,
    compute_estimated_block_contributions,
    compute_true_block_contributions,
    cross_block_cov_shrinkage,
    dominant_block_accuracy,
    edits_to_satisfaction,
    per_block_recovery,
    regret_from_utilities,
    total_explicit_burden,
)


def test_total_explicit_burden():
    d = total_explicit_burden(2, extra=1)
    assert d["clarification_questions"] == 2
    assert d["pairwise_extra"] == 1
    assert d["total_burden"] == 3


def test_per_block_recovery():
    mu = np.array([0.2, -0.1, 0.3, 0.0, 0.1, 0.0])
    beta = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    rec = per_block_recovery(mu, beta)
    assert rec["taste_norm"] == pytest.approx(np.sqrt(0.2**2 + 0.1**2 + 0.3**2))
    assert rec["context_norm"] == pytest.approx(0.1)


def test_cross_block_cov_shrinkage():
    S = np.eye(6)
    d = cross_block_cov_shrinkage(S)
    assert d["cross_block_norm"] == pytest.approx(0.0)
    assert d["shrinkage_fraction"] == pytest.approx(0.0)


def test_contribution_errors_known():
    delta = np.array([1.0, 0.0, 0.0, 0.0, 2.0, 0.0])
    beta = np.array([1.0, 0.0, 0.0, 0.0, 0.5, 0.0])
    mu = np.array([0.8, 0.0, 0.0, 0.0, 0.4, 0.0])
    true = compute_true_block_contributions(delta, beta)
    est = compute_estimated_block_contributions(delta, mu)
    err = contribution_errors(delta, mu, beta)
    assert true["delta_taste"] == pytest.approx(1.0)
    assert true["delta_context"] == pytest.approx(1.0)
    assert err["taste_error"] == pytest.approx(abs(est["delta_taste"] - true["delta_taste"]))
    assert err["context_error"] == pytest.approx(abs(est["delta_context"] - true["delta_context"]))


def test_dominant_block_accuracy_clear():
    delta = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    beta = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    mu = np.array([0.9, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert dominant_block_accuracy(delta, mu, beta, delta=0.1) == "taste"


def test_dominant_block_accuracy_ambiguous_returns_none():
    delta = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    beta = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    mu = np.array([0.5, 0.0, 0.0, 0.0, 0.5, 0.0])
    assert dominant_block_accuracy(delta, mu, beta, delta=0.5) is None


def test_regret():
    utils = np.array([10.0, 20.0, 15.0])
    assert regret_from_utilities(utils, chosen_index=0) == pytest.approx(10.0)
    assert regret_from_utilities(utils, chosen_index=1) == pytest.approx(0.0)


def test_edits_to_satisfaction():
    assert edits_to_satisfaction([False, False, True]) == 3
    assert edits_to_satisfaction([True]) == 1
    assert edits_to_satisfaction([False, False]) == 2
