"""Level-A research architecture utilities.

Frozen Phase-1 likelihood remains untouched.
This module only adds auditable provenance/kind classification,
closed-vocabulary validation, and identifiability diagnostics.

No likelihood weights are changed based on provenance.
No Thompson Sampling is activated in existing Track-A cells.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, List, Optional, Sequence

import numpy as np


# ------------------------------------------------------------------ enums
class Provenance(str, Enum):
    USER_EXPLICIT = "USER_EXPLICIT"
    LLM_PARSED_EXPLICIT = "LLM_PARSED_EXPLICIT"
    LLM_INFERRED = "LLM_INFERRED"
    SYSTEM_GENERATED = "SYSTEM_GENERATED"


class EvidenceKind(str, Enum):
    PREFERENCE = "PREFERENCE"
    FEASIBILITY = "FEASIBILITY"
    NON_LEARNING = "NON_LEARNING"


# fixed six-dim registry (must match preference_features.FEATURE_NAMES)
FIXED_FEATURE_DIMENSIONS = {
    "cuisine_match",
    "fame_touristy",
    "heaviness",
    "travel_min",
    "price_level",
    "queue_wait",
}

# taste block indices [0,1,2], convenience block [3,4,5]
TASTE_BLOCK = {0, 1, 2}
CONVENIENCE_BLOCK = {3, 4, 5}


# ------------------------------------------------------------------ helpers
def attach_evidence_meta(
    record,
    *,
    provenance: Provenance,
    evidence_kind: EvidenceKind,
    source_turn_id: Optional[str] = None,
    parser_version: Optional[str] = None,
    support_text: Optional[str] = None,
    attributed_dims: Optional[Sequence[str]] = None,
):
    """Attach metadata to a record without requiring EvidenceRecord changes.

    Uses __dict__ to avoid Pydantic-like model restrictions.
    """
    record.__dict__["provenance"] = provenance.value
    record.__dict__["evidence_kind"] = evidence_kind.value
    record.__dict__["source_turn_id"] = source_turn_id
    record.__dict__["parser_version"] = parser_version
    record.__dict__["support_text"] = support_text
    record.__dict__["attributed_dims"] = list(attributed_dims or [])
    return record


def classify_evidence(record) -> EvidenceKind:
    """Architectural classification based on current fields + metadata.

    If evidence has provenance SYSTEM_GENERATED it is NON_LEARNING even if
    it has learning=True and not censored_feasibility.
    """
    if not getattr(record, "learning", False):
        return EvidenceKind.NON_LEARNING
    if getattr(record, "censored_feasibility", False):
        return EvidenceKind.FEASIBILITY
    prov = getattr(record, "provenance", None)
    if prov is not None and prov == Provenance.SYSTEM_GENERATED.value:
        return EvidenceKind.NON_LEARNING
    if record.event_type in {"replacement", "clarification_answer", "explicit_critique"}:
        return EvidenceKind.PREFERENCE
    return EvidenceKind.NON_LEARNING


def validate_structured_evidence(
    *,
    attributes: Sequence[str],
    source_text: Optional[str] = None,
    available_entities: Optional[Sequence[str]] = None,
    offered_options: Optional[Sequence[str]] = None,
) -> tuple[bool, Optional[str], Optional[Provenance]]:
    """Closed-vocabulary validation.

    Returns (ok, reason_or_None, downgraded_provenance).
    - If all attrs are in FIXED_FEATURE_DIMENSIONS and at least one has
      textual support, OK.
    - If an attr is not in fixed registry -> reject preference evidence.
    - If attr in registry but no support -> downgrade to LLM_INFERRED and
      note that T1 should keep it NON_LEARNING.
    """
    unknown = [a for a in attributes if a not in FIXED_FEATURE_DIMENSIONS]
    if unknown:
        return False, f"Unsupported dimension(s): {unknown}", None

    supported = False
    if source_text:
        supported = True
    if offered_options:
        for a in attributes:
            if a in offered_options:
                supported = True
    if available_entities:
        pass  # entity presence does not by itself prove dimension attribution

    if not supported:
        return False, (
            "No textual support or offered option for dimension attribution; "
            "downgrade to LLM_INFERRED / NON_LEARNING for T1"
        ), Provenance.LLM_INFERRED

    return True, None, Provenance.LLM_PARSED_EXPLICIT


def compute_pool_identifiability_diagnostics(
    Phi: np.ndarray,
) -> dict:
    """Given candidate-pool feature matrix (n,6) return correlation/eigen info."""
    if Phi.ndim != 2 or Phi.shape[1] != 6:
        raise ValueError("Phi must be a 2D array with 6 columns")
    Phi = np.asarray(Phi, dtype=float)
    n = Phi.shape[0]
    if n < 2:
        corr = np.full((6, 6), np.nan)
        return {
            "corr_matrix": corr,
            "taste_context_corr_block": np.full((3, 3), np.nan),
            "max_abs_offdiag_corr": np.nan,
            "feature_singular_values": np.zeros(6),
            "effective_rank": 0,
            "numerical_rank": 0,
            "condition_number": np.nan,
            "zero_variance_dims": list(np.where(np.std(Phi, axis=0) == 0)[0]),
        }
    std = np.std(Phi, axis=0)
    corr = np.corrcoef(Phi, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    taste_idx = sorted(TASTE_BLOCK)
    conv_idx = sorted(CONVENIENCE_BLOCK)
    block = corr[np.ix_(taste_idx, conv_idx)]
    sv = np.linalg.svd(Phi - np.mean(Phi, axis=0), compute_uv=False)
    if n >= 6 and np.any(sv > 1e-12):
        cond = sv[0] / sv[-1] if sv[-1] > 1e-12 else float("inf")
    else:
        cond = float("inf") if np.max(sv) > 1e-12 else np.nan
    eff_rank = int(np.sum(sv > 1e-10))
    num_rank = int(np.linalg.matrix_rank(Phi - np.mean(Phi, axis=0), tol=1e-10))
    offdiag = corr.copy()
    np.fill_diagonal(offdiag, 0.0)
    max_abs_off = float(np.max(np.abs(offdiag)))
    return {
        "corr_matrix": corr,
        "taste_context_corr_block": block,
        "max_abs_offdiag_corr": max_abs_off,
        "feature_singular_values": sv,
        "effective_rank": eff_rank,
        "numerical_rank": num_rank,
        "condition_number": cond,
        "zero_variance_dims": list(np.where(std == 0)[0]),
    }


def compute_posterior_design_identifiability(
    mu: np.ndarray,
    Sigma: np.ndarray,
    X: np.ndarray,
) -> dict:
    """Diagnostic for choice-only posterior / design.

    H_exact = I + Xᵀ diag(p(1-p)) X, with p = sigmoid(mu @ x_e).
    """
    mu = np.asarray(mu, dtype=float).reshape(-1)
    Sigma = np.asarray(Sigma, dtype=float)
    X = np.asarray(X, dtype=float)
    if Sigma.shape != (6, 6):
        raise ValueError("Sigma must be 6x6")
    if X.ndim == 1:
        X = X[None, :]
    p = 1.0 / (1.0 + np.exp(- (X @ mu)))
    H = np.eye(6) + X.T @ np.diag(p * (1 - p)) @ X
    eig_sigma = np.linalg.eigvalsh(Sigma)
    eig_h = np.linalg.eigvalsh(H)
    G = X.T @ X
    eig_g = np.linalg.eigvalsh(G)
    num_rank = np.linalg.matrix_rank(X, tol=1e-10)
    ratios = eig_sigma / 1.0
    return {
        "eig_sigma": eig_sigma,
        "eig_H_exact": eig_h,
        "eig_raw_Gram": eig_g,
        "numerical_rank_X": num_rank,
        "posterior_variance_ratio": ratios,
        "weakly_informed_directions": np.where(np.isclose(ratios, 1.0, atol=0.1))[0].tolist(),
    }


@dataclass
class ProposalPolicy:
    """Minimal greedy / Thompson proposal hook.

    Default stays GREEDY; TS is opt-in via explicit policy/rng.
    """

    @staticmethod
    def greedy_reference_value(base_score: float, mu: np.ndarray, phi: np.ndarray) -> float:
        return float(base_score + np.dot(mu, phi))

    @staticmethod
    def propose(
        base_scores: Sequence[float],
        mu: np.ndarray,
        Sigma: np.ndarray,
        phi_matrix: np.ndarray,
        policy: str = "greedy",
        rng: Optional[np.random.Generator] = None,
    ) -> tuple[int, Optional[np.ndarray]]:
        """Return (argmax_index, beta_tilde_if_TS_else_None)."""
        mu_arr = np.asarray(mu, dtype=float).reshape(-1)
        Sigma_arr = np.asarray(Sigma, dtype=float)
        phi_arr = np.asarray(phi_matrix, dtype=float)
        if phi_arr.ndim != 2 or phi_arr.shape[1] != 6:
            raise ValueError("phi_matrix must be (n,6)")
        if policy == "greedy":
            values = [
                ProposalPolicy.greedy_reference_value(s, mu_arr, p)
                for s, p in zip(base_scores, phi_arr)
            ]
            return int(np.argmax(values)), None
        if policy == "thompson":
            if rng is None:
                raise ValueError("Thompson policy requires an explicit rng")
            beta_tilde = rng.multivariate_normal(mu_arr, Sigma_arr)
            values = [s + float(np.dot(beta_tilde, p)) for s, p in zip(base_scores, phi_arr)]
            return int(np.argmax(values)), beta_tilde
        raise ValueError(f"Unknown policy: {policy}")
