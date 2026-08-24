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
from collections import Counter
from typing import Sequence

import numpy as np

from evidence import EvidenceRecord

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


def _neg_log_posterior(beta, evidences, lam=LAMBDA, tau=TAU, kappa=KAPPA):
    """Negative log posterior: -Σ loglik + 0.5||β||² (prior N(0,I))."""
    beta_arr = np.asarray(beta, dtype=float)
    loglik_sum = 0.0
    for ev in evidences:
        if not ev.learning:
            continue
        if ev.event_type == "replacement":
            if ev.x_e is None:
                continue
            loglik_sum += loglik_choice(beta_arr, ev.x_e, lam)
        elif ev.event_type == "clarification_answer":
            options = ev.question_options or []
            ans = ev.answer_option or ""
            if len(options) >= 3:
                if ans == options[0]:
                    observed = 0
                elif ans == options[1]:
                    observed = 1
                elif ans == options[2]:
                    observed = 2
                else:
                    observed = 2
            elif len(options) == 2:
                observed = 0 if ans == options[0] else 1 if ans == options[1] else 2
            else:
                observed = 2
            j_T, j_C = 0, 1
            loglik_sum += loglik_prompted(beta_arr, ev.x_e, j_T, j_C,
                                          observed, lam, tau, kappa)
        elif ev.event_type == "explicit_critique":
            if ev.x_e is None:
                continue
            x_arr = np.asarray(ev.x_e, dtype=float)
            j = int(np.argmax(np.abs(x_arr)))
            rho = ev.weight if ev.weight is not None else 1.0
            loglik_sum += loglik_critique(beta_arr, ev.x_e, j, rho, lam, tau)
        # bare_rejection excluded via learning=False

    return -loglik_sum + 0.5 * float(np.dot(beta_arr, beta_arr))


def _numeric_gradient(f, beta, eps=1e-5):
    beta = np.asarray(beta, dtype=float)
    g = np.zeros_like(beta)
    for i in range(beta.size):
        xp = beta.copy()
        xp[i] += eps
        xm = beta.copy()
        xm[i] -= eps
        g[i] = (f(xp) - f(xm)) / (2.0 * eps)
    return g


def _numeric_hessian(f, beta, eps=1e-4):
    beta = np.asarray(beta, dtype=float)
    n = beta.size
    H = np.zeros((n, n))
    for j in range(n):
        xp = beta.copy()
        xp[j] += eps
        xm = beta.copy()
        xm[j] -= eps
        gp = _numeric_gradient(f, xp, eps)
        gm = _numeric_gradient(f, xm, eps)
        H[:, j] = (gp - gm) / (2.0 * eps)
    H = (H + H.T) * 0.5
    return H


def refit_laplace(
    evidences,
    *,
    lam: float = LAMBDA,
    tau: float = TAU,
    kappa: float = KAPPA,
    max_iter: int = 50,
    warn_norm_threshold: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit MAP via damped Newton with backtracking line search,
    then return (mu, Sigma) where Sigma = inv(Hessian at MAP).
    """
    learning_rows = [e for e in evidences if e.learning]
    if not learning_rows:
        return np.zeros(6), np.eye(6)

    event_counts: dict = {}
    for e in learning_rows:
        event_counts[e.event_type] = event_counts.get(e.event_type, 0) + 1

    def f(b):
        return _neg_log_posterior(b, learning_rows, lam, tau, kappa)

    beta = np.zeros(6, dtype=float)

    for it in range(max_iter):
        val = f(beta)
        g = _numeric_gradient(f, beta)
        H = _numeric_hessian(f, beta)
        try:
            step = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            step = np.linalg.solve(H + 1e-6 * np.eye(6), -g)

        improved = False
        alpha = 1.0
        for _ in range(30):
            trial = beta + alpha * step
            if np.isfinite(f(trial)) and f(trial) < val - 1e-12:
                improved = True
                break
            alpha *= 0.5
        if not improved:
            break

        beta = trial
        norm_beta = float(np.linalg.norm(beta))
        if norm_beta > warn_norm_threshold:
            print(
                f"[WARNING] MAP norm {norm_beta:.2f} exceeded {warn_norm_threshold} "
                f"at iter {it}; rows={len(learning_rows)} events={event_counts}"
            )

    H_final = _numeric_hessian(f, beta)
    Sigma = np.linalg.inv(H_final + 1e-9 * np.eye(6))
    mu = beta
    return mu, Sigma


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

    # A4: MAP fit should stay bounded on identical replacement rows
    replacement_rows = []
    for i in range(30):
        replacement_rows.append(
            EvidenceRecord(
                evidence_id=f"replace-{i}",
                thread_id="test-thread",
                ts="2024-01-01T00:00:00",
                event_type="replacement",
                learning=True,
                censored_feasibility=False,
                x_e=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ask_eligible=True,
            )
        )
    mu, Sigma = refit_laplace(replacement_rows)
    print("MAP_mu", np.round(mu, 4))
    print("Sigma_diag", np.round(np.diag(Sigma), 4))
    assert np.linalg.norm(mu) < 20.0, f"MAP diverged: ||mu||={np.linalg.norm(mu):.2f}"

    # mixed-event smoke test
    mixed = replacement_rows[:5]
    for i in range(3):
        mixed.append(
            EvidenceRecord(
                evidence_id=f"clarify-{i}",
                thread_id="test-thread",
                ts="2024-01-01T00:00:00",
                event_type="clarification_answer",
                learning=True,
                censored_feasibility=False,
                x_e=[0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                question_options=["A", "B", "其他"],
                answer_option="A",
                ask_eligible=True,
            )
        )
    mixed.append(
        EvidenceRecord(
            evidence_id="critique-1",
            thread_id="test-thread",
            ts="2024-01-01T00:00:00",
            event_type="explicit_critique",
            learning=True,
            censored_feasibility=False,
            x_e=[0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
            weight=0.5,
            ask_eligible=True,
        )
    )
    mu2, Sigma2 = refit_laplace(mixed)
    print("MIXED_MAP", np.round(mu2, 4))
    assert np.all(np.isfinite(mu2))
    assert np.all(np.isfinite(Sigma2))

    print("All A3+A4 checks passed.")
