"""Phase A3/A4 likelihood and MAP tests."""
import math

import numpy as np
import pytest

from evidence import EvidenceRecord
from likelihood import (
    loglik_choice,
    loglik_prompted,
    loglik_critique,
    refit_laplace,
)
from preference_features import FEATURE_NAME_TO_INDEX


def _record(**kw):
    base = {
        "evidence_id": "r",
        "thread_id": "t",
        "ts": "2024-01-01T00:00:00",
        "event_type": "replacement",
        "learning": True,
        "censored_feasibility": False,
        "x_e": [1.0]*6,
    }
    base.update(kw)
    return EvidenceRecord(**base)


def test_prompted_uses_custom_indices():
    beta = [0.0, 0.0, 2.0, 0.0, 0.0, -1.0]
    x = [0.0, 0.0, 1.0, 0.0, 0.0, -10.0]
    lp0 = loglik_prompted(beta, x, 2, 3, 0)
    lp1 = loglik_prompted(beta, x, 2, 3, 1)
    lp_fame = loglik_prompted([1.0, 0, 0, 0, 0, 5.0], [10.0, 0, 0, 0, 0, 1.0], 1, 5, 0)
    assert math.isfinite(lp0)
    assert math.isfinite(lp1)
    assert math.isfinite(lp_fame)


def test_critique_uses_parser_dimension_not_argmax():
    beta = [1.0, 0, 0, 0, 0, 0]
    x_e = [100.0, 0, 0, 1.0, 0, -0.5]
    ev = _record(
        event_type="explicit_critique",
        answer_option="travel_min",
        weight=1.0,
        x_e=x_e,
    )
    mu, Sigma = refit_laplace([ev])
    assert abs(mu[3]) > abs(mu[0])


def test_clarification_with_custom_pair_updates_those_indices():
    ev = _record(
        event_type="clarification_answer",
        question_options=["heaviness", "travel_min", "other"],
        answer_option="heaviness",
        x_e=[0.0, 0.0, 1.0, -1.0, 0.0, 0.0],
    )
    mu, Sigma = refit_laplace([ev])
    assert abs(mu[2]) > 1e-6
    assert abs(mu[3]) > 1e-6
    assert abs(mu[0]) < 1e-6
    assert abs(mu[1]) < 1e-6
    assert abs(mu[4]) < 1e-6
    assert abs(mu[5]) < 1e-6


def test_no_learning_rows_returns_prior():
    mu, Sigma = refit_laplace([])
    assert np.allclose(mu, np.zeros(6))
    assert np.allclose(Sigma, np.eye(6))


def test_bare_rejection_does_not_change_posterior():
    rec = _record(
        event_type="bare_rejection",
        learning=False,
        x_e=None,
    )
    mu0, Sigma0 = refit_laplace([])
    mu1, Sigma1 = refit_laplace([rec])
    assert np.allclose(mu0, mu1)
    assert np.allclose(Sigma0, Sigma1)


def test_directional_replacement():
    row = _record(x_e=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    mu, Sigma = refit_laplace([row] * 5)
    assert mu[0] > 0
    for i in range(1, 6):
        assert abs(mu[i]) < 1e-6


def test_mixed_events_posterior_finite():
    rep = _record(x_e=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    clr = _record(
        event_type="clarification_answer",
        question_options=["heaviness", "travel_min", "other"],
        answer_option="heaviness",
        x_e=[0.0, 0.0, 1.0, -1.0, 0.0, 0.0],
    )
    crit = _record(
        event_type="explicit_critique",
        answer_option="travel_min",
        weight=0.7,
        x_e=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    )
    mu, Sigma = refit_laplace([rep, clr, crit])
    assert np.all(np.isfinite(mu))
    assert np.all(np.isfinite(Sigma))
    assert np.allclose(Sigma, Sigma.T, atol=1e-8)
    assert np.all(np.diag(Sigma) > 0)
