"""Phase 1 unit ladder for the preference learner math."""
from __future__ import annotations

import numpy as np
import pytest

from evidence import EvidenceRecord
from likelihood import (
    LAMBDA_CHOICE,
    LAMBDA_REPORT,
    _neg_log_posterior,
    _numeric_gradient,
    _numeric_hessian,
    refit_laplace,
)


def _record(x_e, event_type="replacement"):
    return EvidenceRecord(
        evidence_id="test",
        thread_id="test-thread",
        ts="2024-01-01T00:00:00",
        event_type=event_type,
        learning=True,
        censored_feasibility=False,
        x_e=list(x_e),
        ask_eligible=False,
    )


def test_empty_evidence_returns_prior():
    mu, Sigma = refit_laplace([])
    assert np.allclose(mu, np.zeros(6), atol=1e-9)
    assert np.allclose(Sigma, np.eye(6), atol=1e-9)
    max_eig = np.max(np.linalg.eigvalsh(Sigma))
    assert max_eig <= 1.0 + 1e-6


def test_pairwise_probability_symmetry():
    rng = np.random.default_rng(123)
    for _ in range(100):
        beta = rng.normal(size=6)
        x = rng.normal(size=6)
        z = float(np.dot(beta, x))
        p1 = 1.0 / (1.0 + np.exp(-z))
        p2 = 1.0 / (1.0 + np.exp(+z))
        assert np.isclose(p1 + p2, 1.0, atol=1e-12)


def test_single_observation_sign_positive():
    ev = _record([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    mu, Sigma = refit_laplace([ev])
    assert mu[0] > 0.0
    assert np.allclose(mu[1:], np.zeros(5), atol=1e-8)


def test_single_observation_sign_negative():
    ev = _record([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    mu, Sigma = refit_laplace([ev])
    assert mu[0] < 0.0
    assert np.allclose(mu[1:], np.zeros(5), atol=1e-8)


def test_gradient_finite_difference_agreement():
    rng = np.random.default_rng(7)
    beta0 = rng.normal(size=6)
    evs = [_record(rng.normal(size=6)) for _ in range(10)]

    def f(b):
        return _neg_log_posterior(b, evs)

    g1 = _numeric_gradient(f, beta0, eps=1e-5)
    g2 = _numeric_gradient(f, beta0, eps=1e-7)
    rel = np.linalg.norm(g1 - g2) / max(1.0, np.linalg.norm(g2))
    assert rel < 1e-4


def test_hessian_finite_difference_agreement():
    rng = np.random.default_rng(11)
    beta0 = rng.normal(size=6)
    evs = [_record(rng.normal(size=6)) for _ in range(10)]

    def f(b):
        return _neg_log_posterior(b, evs)

    H1 = _numeric_hessian(f, beta0, eps=1e-4)
    H2 = _numeric_hessian(f, beta0, eps=1e-5)
    rel = np.linalg.norm(H1 - H2) / max(1.0, np.linalg.norm(H2))
    assert rel < 1e-3


def test_map_stationarity():
    rng = np.random.default_rng(23)
    evs = []
    for i in range(15):
        evs.append(_record(rng.normal(size=6)))
    mu, _ = refit_laplace(evs)

    def f(b):
        return _neg_log_posterior(b, evs)

    g = _numeric_gradient(f, mu, eps=1e-5)
    assert np.linalg.norm(g) < 1e-3
