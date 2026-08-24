"""
Phase A3 — pure likelihood functions (5.2 / 5.3a / 5.3b).

These functions compute log-likelihood contributions for three evidence
types.  They are intentionally pure (no state, no side effects).

Constants:
    LAMBDA = 0.1
    TAU     = 1.0
    KAPPA   = 0.0
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

LAMBDA = 0.1
TAU = 1.0
KAPPA = 0.0


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        e = math.exp(x)
        return e / (1.0 + e)


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    shifted = logits - np.max(logits)
    exps = np.exp(shifted)
    return exps / exps.sum()


def loglik_choice(
    beta: Sequence[float],
    x_e: Sequence[float],
    lam: float = LAMBDA,
) -> float:
    """
    5.2 replacement / choice likelihood.

    P(replace with item) = lam/2 + (1-lam)*sigmoid(beta @ x_e)

    Returns log(P).
    """
    z = float(np.dot(beta, x_e))
    p = lam * 0.5 + (1.0 - lam) * _sigmoid(z)
    p = max(p, 1e-15)
    return math.log(p)


def loglik_prompted(
    beta: Sequence[float],
    x_e: Sequence[float],
    j_T: int,
    j_C: int,
    observed_o: int,
    lam: float = LAMBDA,
    tau: float = TAU,
    kappa: float = KAPPA,
) -> float:
    """
    5.3a prompted answer likelihood (3-way: j_T, j_C, other).

    P(o) = lam/3 + (1-lam)*softmax_tau(
                    {beta[j_T]*x_e[j_T],
                     beta[j_C]*x_e[j_C],
                     kappa})[observed_o]

    `observed_o` is the index into the 3-element logit vector:
        0 -> j_T, 1 -> j_C, 2 -> other.
    """
    logits = np.array([
        beta[j_T] * x_e[j_T],
        beta[j_C] * x_e[j_C],
        kappa,
    ], dtype=float)
    probs = _softmax(logits / tau)
    p = lam / 3.0 + (1.0 - lam) * probs[observed_o]
    p = max(p, 1e-15)
    return math.log(p)


def loglik_critique(
    beta: Sequence[float],
    x_e: Sequence[float],
    j: int,
    rho: float,
    lam: float = LAMBDA,
    tau: float = TAU,
) -> float:
    """
    5.3b spontaneous critique likelihood.

    Conditional / pseudo‑observation — conditioned on the parser having
    already observed that the user explicitly points at dimension `j`.
    It is treated as a noisy sign observation that dimension `j` supports
    the current action.

    P(assert j) = lam/2 + (1-lam)*sigmoid(beta[j] * x_e[j] / tau)

    Returns rho * log(P); thus rho=0 contributes exactly 0.
    """
    z = float(beta[j] * x_e[j]) / tau
    p = lam * 0.5 + (1.0 - lam) * _sigmoid(z)
    p = max(p, 1e-15)
    return rho * math.log(p)


if __name__ == "__main__":
    # Basic numerical sanity checks
    big = 50.0
    s_pos = _sigmoid(big)
    s_neg = _sigmoid(-big)
    assert 0.0 < s_pos < 1.0
    assert 0.0 < s_neg < 1.0
    assert math.isclose(s_pos + s_neg, 1.0, abs_tol=1e-12)

    # loglik_choice with extreme values should not overflow
    beta = [1.0, -2.0, 0.5, 0.0, 0.0, 0.0]
    x_e = [10.0, -10.0, 5.0, 0.0, 0.0, 0.0]
    lc = loglik_choice(beta, x_e)          # beta dot = 10 - (-20) + 2.5 = 32.5
    assert math.isfinite(lc)

    # prompted: probabilities sum to 1
    probs = []
    for o in range(3):
        lp = loglik_prompted(beta, x_e, j_T=0, j_C=1, observed_o=o)
        probs.append(math.exp(lp))
    total = sum(probs)
    print("P_sum", total)
    assert math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-12)

    # critique: rho=0 -> contribution 0
    lcrit = loglik_critique(beta, x_e, j=1, rho=0.0)
    assert math.isclose(lcrit, 0.0, abs_tol=1e-15)
    print("rho0_loglik", lcrit)

    # extreme softmax (no overflow)
    logits_ext = np.array([1000.0, -1000.0, 0.0])
    probs_ext = _softmax(logits_ext)
    assert np.all(np.isfinite(probs_ext))

    print("All A3 sanity checks passed.")
