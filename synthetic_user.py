"""Synthetic user generator for Phase D simulation.

Scope: only the generator. No experiment arm, no evaluation loop.
"""

from __future__ import annotations

from typing import Optional

from dataclasses import dataclass

import numpy as np

from config import P_CRIT_GRID, SIGMA_ITEM_GRID


@dataclass
class SyntheticDecisionEvent:
    phi: np.ndarray          # (N,6) model coordinates
    s0_tilde: np.ndarray     # (N,)
    item_residual: np.ndarray  # (N,)
    true_best_index: int
    system_choice_index: int
    current_index: int


@dataclass
class SyntheticTripWorld:
    beta_star: np.ndarray          # (6,)
    events: list[SyntheticDecisionEvent]

    def get_event(self, idx: int) -> SyntheticDecisionEvent:
        return self.events[idx]


def generate_trip_world(
    *,
    n_events: int,
    n_candidates: int,
    beta_star: Optional[np.ndarray],
    residual_multiplier: float,
    rng: np.random.Generator,
) -> SyntheticTripWorld:
    """Create a multi-event synthetic world shared across arms.

    Each event has N>=8 candidates with mixed taste/context variation,
    an S0_tilde baseline, and fixed item residuals.  beta_star is drawn
    once for the whole world.
    """
    if beta_star is None:
        beta_star = rng.normal(size=6)

    events: list[SyntheticDecisionEvent] = []
    for _ in range(n_events):
        N = max(8, n_candidates)
        phi = rng.normal(size=(N, 6))
        # standardize each event matrix roughly to mean 0 / std 1
        phi = (phi - phi.mean(axis=0)) / (phi.std(axis=0) + 1e-8)
        s0_tilde = rng.normal(size=N)

        # signal scale for residual multiplier
        systematic = s0_tilde + phi @ beta_star
        v_signal = float(np.var(systematic))
        sigma2_item = residual_multiplier * v_signal
        residual = rng.normal(scale=np.sqrt(sigma2_item), size=N)

        utilities = systematic + residual
        true_best = int(np.argmax(utilities))
        # system choice uses only S0_tilde (μ=0 placeholder)
        system_choice = int(np.argmax(s0_tilde))
        current_index = system_choice
        events.append(
            SyntheticDecisionEvent(
                phi=phi,
                s0_tilde=s0_tilde,
                item_residual=residual,
                true_best_index=true_best,
                system_choice_index=system_choice,
                current_index=current_index,
            )
        )
    return SyntheticTripWorld(beta_star=beta_star, events=events)


def generate_synthetic_user(
    feature_vectors: np.ndarray,
    current_index: int,
    *,
    beta_star: Optional[np.ndarray] = None,
    sigma_item_variance: float = 0.0,
    p_crit: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """Generate one synthetic user revision.

    Args:
        feature_vectors: (n, 6) array of model coordinates (phi_model).
        current_index: index of the current item X being replaced.
        beta_star: true preference vector. If None, sampled N(0, I_6).
        sigma_item_variance: σ²_item (variance) for item residual.
        p_crit: probability the user spontaneously emits a critique.
        rng: numpy random generator.

    Returns:
        dict with keys:
            beta_star (np.ndarray shape (6,)),
            chosen_index (int),
            utilities (np.ndarray length n),
            epsilon (np.ndarray length n),
            critique_emitted (bool),
            critique_dim (int|None),
    """
    if rng is None:
        rng = np.random.default_rng()

    if beta_star is None:
        beta_star = rng.normal(size=6)

    n = feature_vectors.shape[0]
    if not (0 <= current_index < n):
        raise ValueError(f"current_index out of range: {current_index}")

    epsilon = rng.normal(scale=np.sqrt(sigma_item_variance), size=n)
    utilities = feature_vectors @ beta_star + epsilon

    alt_indices = [i for i in range(n) if i != current_index]
    chosen_index = max(alt_indices, key=lambda i: utilities[i])

    critique_emitted = rng.random() < p_crit
    critique_dim = None
    if critique_emitted:
        delta = beta_star * (feature_vectors[chosen_index] - feature_vectors[current_index])
        pos_delta = delta > 0
        if np.any(pos_delta):
            critique_dim = int(np.argmax(np.where(pos_delta, delta, -np.inf)))
        else:
            critique_dim = None

    delta_phi = feature_vectors[chosen_index] - feature_vectors[current_index]
    return {
        "beta_star": beta_star,
        "chosen_index": chosen_index,
        "utilities": utilities,
        "epsilon": epsilon,
        "critique_emitted": critique_emitted,
        "critique_dim": critique_dim,
        "delta_phi": delta_phi.tolist(),
    }
