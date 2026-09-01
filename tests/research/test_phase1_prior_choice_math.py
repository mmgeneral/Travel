"""Phase‑1 prior / choice‑math tests.

These tests verify the frozen choice likelihood,
independent closed‑form derivative oracles,
and the Sigma <= I invariant for choice‑only logs.
"""
from __future__ import annotations

import numpy as np
import pytest

from evidence import EvidenceRecord
from likelihood import (
    _numeric_gradient,
    _numeric_hessian,
    prob_choice,
    refit_laplace,
)


def _record(x_e, censored=False):
    return EvidenceRecord(
        evidence_id="phase1-test",
        thread_id="phase1-thread",
        ts="2024-01-01T00:00:00",
        event_type="replacement",
        learning=True,
        censored_feasibility=censored,
        x_e=list(np.asarray(x_e, dtype=float)),
        ask_eligible=False,
    )


def _exact_grad(beta, evs):
    """Closed‑form gradient for choice‑only evidence:
    f(β) = 0.5||β||² + Σ log(1+exp(-β·x)).
    ∇f = β − Σ sigmoid(-β·x)·x
    """
    beta = np.asarray(beta, dtype=float)
    grad = beta.copy()
    for ev in evs:
        x = np.asarray(ev.x_e, dtype=float)
        z = float(beta @ x)
        grad -= (1.0 / (1.0 + np.exp(z))) * x
    return grad


def _exact_hess(beta, evs):
    """Closed‑form Hessian for choice‑only evidence:
    H = I + Σ p_i(1-p_i) x_i x_i^T,  p_i = sigmoid(B·x).
    """
    beta = np.asarray(beta, dtype=float)
    H = np.eye(6)
    for ev in evs:
        x = np.asarray(ev.x_e, dtype=float)
        z = float(beta @ x)
        s = 1.0 / (1.0 + np.exp(-z))
        H += s * (1.0 - s) * np.outer(x, x)
    return H


def test_empty_evidence_returns_prior():
    mu, Sigma = refit_laplace([])
    assert np.allclose(mu, np.zeros(6), atol=1e-9)
    assert np.allclose(Sigma, np.eye(6), atol=1e-9)
    max_eig = np.max(np.linalg.eigvalsh(Sigma))
    assert max_eig <= 1.0 + 1e-6


def test_all_censored_evidence_returns_prior():
    ev = _record([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], censored=True)
    mu, Sigma = refit_laplace([ev])
    assert np.allclose(mu, np.zeros(6), atol=1e-9)
    assert np.allclose(Sigma, np.eye(6), atol=1e-9)


def test_single_positive_e1():
    ev = _record([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    mu, _ = refit_laplace([ev])
    assert mu[0] > 0.0
    assert np.allclose(mu[1:], np.zeros(5), atol=1e-6)


def test_single_negative_e1():
    ev = _record([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    mu, _ = refit_laplace([ev])
    assert mu[0] < 0.0
    assert np.allclose(mu[1:], np.zeros(5), atol=1e-6)


def test_probability_symmetry():
    rng = np.random.default_rng(123)
    for _ in range(100):
        beta = rng.normal(size=6)
        x = rng.normal(size=6)
        p1 = prob_choice(beta, x)
        p2 = prob_choice(beta, -x)
        assert np.isclose(p1 + p2, 1.0, atol=1e-12)


def test_gradient_matches_closed_form():
    rng = np.random.default_rng(7)
    beta0 = rng.normal(size=6)
    evs = [_record(rng.normal(size=6)) for _ in range(10)]

    def f(b):
        # choice-only objective: 0.5||β||² + Σ log(1+exp(-β·x))
        b = np.asarray(b, dtype=float)
        ll = np.sum(
            np.logaddexp(0.0, -np.array([float(b @ np.asarray(e.x_e)) for e in evs]))
        )
        return 0.5 * float(b @ b) + ll

    g_num = _numeric_gradient(f, beta0, eps=1e-5)
    g_exact = _exact_grad(beta0, evs)
    rel = np.linalg.norm(g_num - g_exact) / max(1e-12, np.linalg.norm(g_exact))
    assert rel < 1e-4


def test_hessian_matches_closed_form():
    rng = np.random.default_rng(11)
    beta0 = rng.normal(size=6)
    evs = [_record(rng.normal(size=6)) for _ in range(10)]

    def f(b):
        b = np.asarray(b, dtype=float)
        ll = np.sum(
            np.logaddexp(0.0, -np.array([float(b @ np.asarray(e.x_e)) for e in evs]))
        )
        return 0.5 * float(b @ b) + ll

    H_num = _numeric_hessian(f, beta0, eps=1e-4)
    H_exact = _exact_hess(beta0, evs)
    rel = np.linalg.norm(H_num - H_exact) / max(1e-12, np.linalg.norm(H_exact))
    assert rel < 1e-3


def test_map_stationarity_choice_only():
    rng = np.random.default_rng(23)
    evs = [_record(rng.normal(size=6)) for _ in range(20)]
    mu, _ = refit_laplace(evs)
    grad_exact = _exact_grad(mu, evs)
    assert np.linalg.norm(grad_exact) < 1e-3


def test_sigma_bounded_by_identity_choice_only():
    rng = np.random.default_rng(42)
    evs = [_record(rng.normal(size=6)) for _ in range(20)]
    _, Sigma = refit_laplace(evs)
    max_eig = np.max(np.linalg.eigvalsh(Sigma))
    assert max_eig <= 1.0 + 1e-6
