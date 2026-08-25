"""Phase D.3 measurement functions.

Implements:
- prompted clarification question count (explicit burden)
- per-block recovery norms
- cross-block covariance shrinkage diagnostics
- block contribution recovery (true vs estimated)
- dominant-block accuracy (only when clearly separated)
- regret relative to β*-optimal under the same candidate set
- edits-to-satisfaction helper
"""

from __future__ import annotations

import numpy as np

TASTE_IDX = slice(0, 3)
CONTEXT_IDX = slice(3, 6)


def total_explicit_burden(question_count: int, pairwise_extra: int = 0) -> dict:
    """Return prompted clarification question count plus pairwise extra burden."""
    return {
        "clarification_questions": int(question_count),
        "pairwise_extra": int(pairwise_extra),
        "total_burden": int(question_count) + int(pairwise_extra),
    }


def per_block_recovery(mu_est: np.ndarray, beta_star: np.ndarray) -> dict:
    """Return L2 norms of (μ_T − w_T*) and (μ_θ − θ*)."""
    mu = np.asarray(mu_est, dtype=float).reshape(-1)
    beta = np.asarray(beta_star, dtype=float).reshape(-1)
    return {
        "taste_norm": float(np.linalg.norm(mu[TASTE_IDX] - beta[TASTE_IDX])),
        "context_norm": float(np.linalg.norm(mu[CONTEXT_IDX] - beta[CONTEXT_IDX])),
    }


def cross_block_cov_shrinkage(Sigma: np.ndarray) -> dict:
    """Return cross-block covariance magnitude and fraction of total variance."""
    S = np.asarray(Sigma, dtype=float)
    if S.shape != (6, 6):
        raise ValueError("Sigma must be 6x6")
    block = S[TASTE_IDX, CONTEXT_IDX]
    total_norm = float(np.sqrt(np.sum(S**2)))
    cross_norm = float(np.sqrt(np.sum(block**2)))
    return {
        "cross_block_norm": cross_norm,
        "total_norm": total_norm,
        "shrinkage_fraction": cross_norm / (total_norm + 1e-12),
    }


def compute_true_block_contributions(delta_phi: np.ndarray, beta_star: np.ndarray) -> dict:
    d = np.asarray(delta_phi, dtype=float).reshape(-1)
    beta = np.asarray(beta_star, dtype=float).reshape(-1)
    return {
        "delta_taste": float(np.dot(beta[TASTE_IDX], d[TASTE_IDX])),
        "delta_context": float(np.dot(beta[CONTEXT_IDX], d[CONTEXT_IDX])),
        "delta_total": float(np.dot(beta, d)),
    }


def compute_estimated_block_contributions(delta_phi: np.ndarray, mu_est: np.ndarray) -> dict:
    d = np.asarray(delta_phi, dtype=float).reshape(-1)
    mu = np.asarray(mu_est, dtype=float).reshape(-1)
    return {
        "delta_taste": float(np.dot(mu[TASTE_IDX], d[TASTE_IDX])),
        "delta_context": float(np.dot(mu[CONTEXT_IDX], d[CONTEXT_IDX])),
        "delta_total": float(np.dot(mu, d)),
    }


def contribution_errors(delta_phi: np.ndarray, mu_est: np.ndarray, beta_star: np.ndarray) -> dict:
    true = compute_true_block_contributions(delta_phi, beta_star)
    est = compute_estimated_block_contributions(delta_phi, mu_est)
    return {
        "taste_error": abs(est["delta_taste"] - true["delta_taste"]),
        "context_error": abs(est["delta_context"] - true["delta_context"]),
        "total_error": abs(est["delta_total"] - true["delta_total"]),
    }


def dominant_block_accuracy(
    delta_phi: np.ndarray,
    mu_est: np.ndarray,
    beta_star: np.ndarray,
    delta: float = 0.1,
) -> str | None:
    """Return 'taste', 'context', or None when blocks are not clearly separated."""
    true = compute_true_block_contributions(delta_phi, beta_star)
    diff = abs(true["delta_taste"]) - abs(true["delta_context"])
    if abs(diff) <= delta:
        return None
    est = compute_estimated_block_contributions(delta_phi, mu_est)
    if diff > 0:
        return "taste" if abs(est["delta_taste"]) > abs(est["delta_context"]) else "context"
    return "context" if abs(est["delta_context"]) > abs(est["delta_taste"]) else "taste"


def regret_from_utilities(utilities: np.ndarray, chosen_index: int) -> float:
    """Regret = max_i U*(i) − U*(chosen)."""
    u = np.asarray(utilities, dtype=float)
    best = float(np.max(u))
    chosen = float(u[chosen_index])
    return max(0.0, best - chosen)


def edits_to_satisfaction(edit_list: list[bool]) -> int:
    """Count edits until the first True (satisfied) flag."""
    for i, satisfied in enumerate(edit_list):
        if satisfied:
            return i + 1
    return len(edit_list)
