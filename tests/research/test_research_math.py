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
    prob_choice,
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
        p1 = prob_choice(beta, x)
        p2 = prob_choice(beta, -x)
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


def _choice_only_evs(rng, n=10):
    return [_record(rng.normal(size=6)) for _ in range(n)]


def _exact_grad(beta, evs):
    beta = np.asarray(beta, dtype=float)
    grad = beta.copy()
    for ev in evs:
        x = np.asarray(ev.x_e, dtype=float)
        z = float(beta @ x)
        grad -= (1.0 / (1.0 + np.exp(z))) * x
    return grad


def _exact_hess(beta, evs):
    beta = np.asarray(beta, dtype=float)
    H = np.eye(6)
    for ev in evs:
        x = np.asarray(ev.x_e, dtype=float)
        z = float(beta @ x)
        s = 1.0 / (1.0 + np.exp(-z))
        H += s * (1.0 - s) * np.outer(x, x)
    return H


def test_gradient_matches_analytic_choice():
    rng = np.random.default_rng(7)
    beta0 = rng.normal(size=6)
    evs = _choice_only_evs(rng, 10)

    def f(b):
        return _neg_log_posterior(b, evs)

    g_num = _numeric_gradient(f, beta0, eps=1e-5)
    g_exact = _exact_grad(beta0, evs)
    assert np.allclose(g_num, g_exact, rtol=1e-4, atol=1e-5)


def test_hessian_matches_analytic_choice():
    rng = np.random.default_rng(11)
    beta0 = rng.normal(size=6)
    evs = _choice_only_evs(rng, 10)

    def f(b):
        return _neg_log_posterior(b, evs)

    H_num = _numeric_hessian(f, beta0, eps=1e-4)
    H_exact = _exact_hess(beta0, evs)
    assert np.allclose(H_num, H_exact, rtol=1e-3, atol=1e-3)


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


def test_only_censored_evidence_returns_prior():
    ev = EvidenceRecord(
        evidence_id="censored",
        thread_id="thread",
        ts="2024-01-01T00:00:00",
        event_type="replacement",
        learning=True,
        censored_feasibility=True,
        x_e=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ask_eligible=False,
    )
    mu, Sigma = refit_laplace([ev])
    assert np.allclose(mu, np.zeros(6), atol=1e-9)
    assert np.allclose(Sigma, np.eye(6), atol=1e-9)


def test_only_choice_sigma_bounded_by_I():
    rng = np.random.default_rng(42)
    evs = _choice_only_evs(rng, 20)
    mu, Sigma = refit_laplace(evs)
    max_eig = np.max(np.linalg.eigvalsh(Sigma))
    assert max_eig <= 1.0 + 1e-6


def test_hessian_min_eigen_ge_one():
    rng = np.random.default_rng(101)
    evs = _choice_only_evs(rng, 20)
    mu, _ = refit_laplace(evs)
    H_exact = _exact_hess(mu, evs)
    min_eig = np.min(np.linalg.eigvalsh(H_exact))
    assert min_eig >= 1.0 - 1e-6


def test_map_exact_gradient_small():
    rng = np.random.default_rng(24)
    evs = _choice_only_evs(rng, 20)
    mu, _ = refit_laplace(evs)
    grad = _exact_grad(mu, evs)
    assert np.linalg.norm(grad) < 1e-3
