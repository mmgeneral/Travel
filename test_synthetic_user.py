"""Unit tests for Phase D.1 synthetic user generator."""
import numpy as np
import pytest

from synthetic_user import generate_synthetic_user
from config import SIGMA_ITEM_GRID, P_CRIT_GRID


def test_grid_constants_exist():
    assert SIGMA_ITEM_GRID == (0.0, 0.25, 1.0, 4.0)
    assert P_CRIT_GRID == (0.0, 0.3, 0.6)


def test_deterministic_with_seed():
    features = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    ])
    rng = np.random.default_rng(42)
    out1 = generate_synthetic_user(features, current_index=0, rng=rng,
                                   sigma_item_variance=0.0, p_crit=0.0)
    rng2 = np.random.default_rng(42)
    out2 = generate_synthetic_user(features, current_index=0, rng=rng2,
                                   sigma_item_variance=0.0, p_crit=0.0)
    assert np.array_equal(out1["beta_star"], out2["beta_star"])
    assert out1["chosen_index"] == out2["chosen_index"]


def test_chosen_is_not_current():
    features = np.eye(6)[:5]
    out = generate_synthetic_user(features, current_index=0,
                                  sigma_item_variance=0.0, p_crit=0.0,
                                  rng=np.random.default_rng(0))
    assert out["chosen_index"] != 0


def test_trip_world_with_residual_and_s0():
    from synthetic_user import generate_trip_world, SyntheticTripWorld
    rng = np.random.default_rng(11)
    world = generate_trip_world(n_events=3, n_candidates=8,
                                beta_star=None,
                                residual_multiplier=1.0,
                                rng=rng)
    assert isinstance(world, SyntheticTripWorld)
    assert len(world.events) == 3
    assert world.events[0].phi.shape == (8, 6)
    assert world.events[0].true_best_index != world.events[0].system_choice_index


def test_critique_only_when_positive_delta():
    features = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    ])
    out = generate_synthetic_user(features, current_index=0,
                                  beta_star=np.array([1.0, 0.5, 0.0, 0.0, 0.0, 0.0]),
                                  sigma_item_variance=0.0, p_crit=1.0,
                                  rng=np.random.default_rng(1))
    assert out["chosen_index"] == 1
    assert out["critique_dim"] == 0


def test_bare_rejection_no_critique_when_p_crit_zero():
    features = np.eye(6)[:3]
    out = generate_synthetic_user(features, current_index=0,
                                  sigma_item_variance=0.0, p_crit=0.0,
                                  rng=np.random.default_rng(5))
    assert out["critique_emitted"] is False
    assert out["critique_dim"] is None
