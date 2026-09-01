"""Level-A architecture diagnostic tests."""
from __future__ import annotations

import numpy as np
import pytest

from research_architecture import (
    Provenance,
    EvidenceKind,
    classify_evidence,
    validate_structured_evidence,
    compute_pool_identifiability_diagnostics,
    compute_posterior_design_identifiability,
    ProposalPolicy,
)
from config import LAMBDA_CHOICE, LAMBDA_REPORT
from evidence import EvidenceRecord


class _FakeRecord:
    def __init__(self, *, event_type="replacement", learning=True, censored_feasibility=False,
                 provenance=None, evidence_kind=None):
        self.event_type = event_type
        self.learning = learning
        self.censored_feasibility = censored_feasibility
        self.provenance = provenance
        self.evidence_kind = evidence_kind


def test_provenance_enum_values():
    assert Provenance.USER_EXPLICIT.value == "USER_EXPLICIT"
    assert Provenance.LLM_PARSED_EXPLICIT.value == "LLM_PARSED_EXPLICIT"
    assert Provenance.LLM_INFERRED.value == "LLM_INFERRED"
    assert Provenance.SYSTEM_GENERATED.value == "SYSTEM_GENERATED"


def test_system_generated_non_learning():
    rec = _FakeRecord(event_type="replacement", provenance=Provenance.SYSTEM_GENERATED.value)
    assert classify_evidence(rec) == EvidenceKind.NON_LEARNING


def test_feasibility_non_learning():
    rec = _FakeRecord(event_type="replacement", censored_feasibility=True)
    assert classify_evidence(rec) == EvidenceKind.FEASIBILITY


def test_evidence_record_accepts_provenance_and_kind():
    rec = EvidenceRecord(
        evidence_id="e1",
        thread_id="t1",
        ts="2024-01-01T00:00:00",
        event_type="replacement",
        learning=True,
        censored_feasibility=False,
        x_e=[0.1, 0, 0, 0, 0, 0],
        ask_eligible=False,
        provenance=Provenance.USER_EXPLICIT.value,
        evidence_kind=EvidenceKind.PREFERENCE.value,
    )
    assert rec.provenance == Provenance.USER_EXPLICIT.value
    assert rec.evidence_kind == EvidenceKind.PREFERENCE.value


def test_unknown_dim_rejected():
    ok, reason, prov = validate_structured_evidence(
        attributes=["ambiance"],
        source_text="because it is cozy",
    )
    assert not ok
    assert "Unsupported dimension" in reason


def test_supported_explicit_dim_accepted():
    ok, _, prov = validate_structured_evidence(
        attributes=["travel_min"],
        source_text="because it is closer",
    )
    assert ok
    assert prov == Provenance.LLM_PARSED_EXPLICIT


def test_pool_diag_6x6():
    rng = np.random.default_rng(0)
    Phi = rng.normal(size=(30, 6))
    out = compute_pool_identifiability_diagnostics(Phi)
    assert out["corr_matrix"].shape == (6, 6)


def test_constant_feature_no_nan():
    Phi = np.column_stack([np.ones(30), np.linspace(0, 1, 30), np.random.randn(30, 4)])
    out = compute_pool_identifiability_diagnostics(Phi)
    assert not np.isnan(out["corr_matrix"]).any()


def test_posterior_diag_preserves_sigma():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(20, 6))
    mu = rng.normal(size=6)
    Sigma = rng.dirichlet(np.ones(6), size=1)[0] * np.eye(6)
    out = compute_posterior_design_identifiability(mu, Sigma, X)
    assert out["eig_sigma"].shape == (6,)


def test_ts_same_seed_reproducible():
    rng1 = np.random.default_rng(7)
    rng2 = np.random.default_rng(7)
    base = [0.0, 0.0, 0.0]
    phi = np.array([[1.0, 0, 0, 0, 0, 0],
                    [0.0, 1.0, 0, 0, 0, 0],
                    [0.0, 0.0, 1.0, 0, 0, 0]])
    mu = np.zeros(6)
    Sigma = np.eye(6)
    idx1, b1 = ProposalPolicy.propose(base, mu, Sigma, phi, "thompson", rng1)
    idx2, b2 = ProposalPolicy.propose(base, mu, Sigma, phi, "thompson", rng2)
    assert b1 is not None
    assert b2 is not None
    assert np.allclose(b1, b2)
    assert idx1 == idx2


class _CountingRng:
    """Counting wrapper that records how many times multivariate_normal is called."""
    def __init__(self, inner, beta_value):
        self.inner = inner
        self.beta_value = np.asarray(beta_value, dtype=float)
        self.count = 0

    def multivariate_normal(self, mean, cov):
        self.count += 1
        return self.beta_value


def test_ts_single_beta_per_decision():
    base = [0.0, 0.0, 0.0]
    phi = np.array([[1.0, 0, 0, 0, 0, 0],
                    [0.0, 1.0, 0, 0, 0, 0],
                    [0.0, 0.0, 1.0, 0, 0, 0]])
    mu = np.zeros(6)
    Sigma = np.eye(6)
    beta_val = np.array([1.0, -1.0, 0.5, 0.0, 0.0, 0.0])
    counting = _CountingRng(np.random.default_rng(7), beta_val)
    idx, returned_beta = ProposalPolicy.propose(base, mu, Sigma, phi, "thompson", counting)
    assert counting.count == 1
    assert np.allclose(returned_beta, beta_val)
    vals = [base[i] + float(np.dot(beta_val, phi[i])) for i in range(3)]
    assert idx == int(np.argmax(vals))


def test_ts_sigma_zero_equals_greedy_preference():
    base = [1.0, 2.0, 3.0]
    phi = np.array([[0.1, 0.2, 0.3, 0.0, 0.0, 0.0],
                    [-0.2, 0.1, 0.0, 0.5, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.7, 0.1]])
    mu = np.array([0.3, -0.2, 0.1, 0.4, -0.6, 0.2])
    Sigma = np.zeros((6, 6))
    rng = np.random.default_rng(9)
    idx_ts, _ = ProposalPolicy.propose(base, mu, Sigma, phi, "thompson", rng)
    idx_greedy, _ = ProposalPolicy.propose(base, mu, Sigma, phi, "greedy")
    assert idx_ts == idx_greedy


def test_ts_does_not_mutate_learner_state():
    mu = np.array([0.1, 0.2, -0.1, 0.0, 0.0, 0.0])
    Sigma = np.eye(6)
    mu_copy = mu.copy()
    Sigma_copy = Sigma.copy()
    rng = np.random.default_rng(13)
    ProposalPolicy.propose([0.0] * 3, mu, Sigma, np.eye(6), "thompson", rng)
    assert np.allclose(mu, mu_copy)
    assert np.allclose(Sigma, Sigma_copy)


def test_greedy_reproduces_deterministic():
    base = [10.0, 5.0, 0.0]
    phi = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 2.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    mu = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    Sigma = np.eye(6)
    idx, _ = ProposalPolicy.propose(base, mu, Sigma, phi, "greedy")
    assert idx == 0  # 10+1 > 5+2 > 0+1
