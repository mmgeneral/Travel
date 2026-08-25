"""Phase D.2 regression: real C1/C2 trajectory equivalence and arm-order invariance."""
import numpy as np
import pytest
import config
from types import SimpleNamespace

from synthetic_user import generate_trip_world
from experiment_arms import run_episode


def _make_world():
    rng = np.random.default_rng(2026)
    return generate_trip_world(
        n_events=3,
        n_candidates=8,
        beta_star=None,
        residual_multiplier=0.0,
        rng=rng,
        world_seed=2026,
    )


def _run_arm(arm, *, force_ask=False, p_crit=0.0, c_int=0.05, world=None, seed=0):
    if world is None:
        world = _make_world()
    arm_rng = np.random.default_rng(seed)
    st = run_episode(
        world=world,
        arm=arm,
        p_crit=p_crit,
        c_int=c_int,
        rng=arm_rng,
        force_ask=force_ask,
        evoi_mc_draws=20,
    )
    return st


def test_c1_c2_trajectories_bit_identical_when_force_ask():
    world = _make_world()
    st1 = _run_arm("C1", force_ask=True, world=world, seed=11)
    st2 = _run_arm("C2", force_ask=True, world=world, seed=11)

    assert st1.proposal_trace == st2.proposal_trace
    assert st1.revision_count == st2.revision_count
    assert st1.question_trace == st2.question_trace
    assert st1.answer_trace == st2.answer_trace
    assert st1.evidence_log == st2.evidence_log
    assert len(st1.posterior_trace) == len(st2.posterior_trace)
    for p1, p2 in zip(st1.posterior_trace, st2.posterior_trace):
        assert p1["mu"] == p2["mu"]
    assert np.allclose(st1.mu, st2.mu)
    assert np.allclose(st1.Sigma, st2.Sigma)
    assert st1.clarification_count == st2.clarification_count
    assert st1.regret_trace == st2.regret_trace


def test_c2_evoi_does_not_depend_on_beta_star():
    """Anti‑oracle regression: compute_evoi_for_questions is called without any
    beta_star parameter, hence changing beta_star cannot directly affect C2's
    EVOI calculation.
    """
    import experiment_arms
    original_fn = experiment_arms.compute_evoi_for_questions

    def wrapper(*args, **kwargs):
        assert "beta_star" not in kwargs
        return original_fn(*args, **kwargs)

    experiment_arms.compute_evoi_for_questions = wrapper
    try:
        world = _make_world()
        _run_arm("C2", force_ask=False, world=world, seed=21, p_crit=0.0)
    finally:
        experiment_arms.compute_evoi_for_questions = original_fn


def test_arm_order_invariance():
    world = _make_world()
    order1 = run_episode(world=world, arm="C0", p_crit=0.0, c_int=0.05,
                         rng=np.random.default_rng(1), evoi_mc_draws=20)
    order2 = run_episode(world=world, arm="C2", p_crit=0.0, c_int=0.05,
                         rng=np.random.default_rng(2), evoi_mc_draws=20)
    # run reverse order
    world2 = _make_world()
    order1b = run_episode(world=world2, arm="C2", p_crit=0.0, c_int=0.05,
                          rng=np.random.default_rng(2), evoi_mc_draws=20)
    world3 = _make_world()
    order0b = run_episode(world=world3, arm="C0", p_crit=0.0, c_int=0.05,
                          rng=np.random.default_rng(1), evoi_mc_draws=20)
    assert order1.proposal_trace == order0b.proposal_trace
    assert order2.proposal_trace == order1b.proposal_trace
    assert order1.evidence_log == order0b.evidence_log
    assert order2.evidence_log == order1b.evidence_log


def test_c3_actually_pairs():
    world = _make_world()
    st = run_episode(world=world, arm="C3", p_crit=0.0, c_int=0.05,
                     rng=np.random.default_rng(4), evoi_mc_draws=20)
    assert st.pairwise_count >= 1


def test_c4_stays_prior():
    world = _make_world()
    st = run_episode(world=world, arm="C4", p_crit=0.0, c_int=0.05,
                     rng=np.random.default_rng(4), evoi_mc_draws=20)
    assert np.allclose(st.mu, np.zeros(6))
    assert np.allclose(st.Sigma, np.eye(6))
    assert len(st.evidence_log) == 0
