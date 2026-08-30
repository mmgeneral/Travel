"""Phase D.2 regression: real C1/C2 trajectory equivalence and arm-order invariance."""
import copy
import numpy as np
import pytest

from synthetic_user import generate_trip_world
from experiment_arms import run_episode, _compute_contender


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


def test_gate_continue_no_ask(monkeypatch):
    import experiment_arms
    from preference_features import FEATURE_NAMES
    calls = []

    def fake_gate(**kwargs):
        calls.append(kwargs)
        return {"action": "continue"}

    monkeypatch.setattr(experiment_arms, "evaluate_gate", fake_gate)

    # force at least one eligible question
    def fake_generate(Sigma, L_j, x_e):
        return [{
            "j_T": 0,
            "j_C": 3,
            "question_options": [
                FEATURE_NAMES[0],
                FEATURE_NAMES[3],
                "other",
            ],
        }], True

    monkeypatch.setattr(experiment_arms, "generate_cross_block_questions", fake_generate)

    # ensure the contender set is larger than 1 so evaluate_gate is reached
    def fake_contender(*args, **kwargs):
        return np.array([0, 1], dtype=int), np.ones(6)
    monkeypatch.setattr(experiment_arms, "_compute_contender", fake_contender)

    world = _make_world()
    st = _run_arm("C2", force_ask=False, world=world, seed=51)
    assert st.clarification_count == 0
    assert len(calls) > 0


def test_c2_can_ask_fewer_than_c1():
    world = _make_world()
    st1 = _run_arm("C1", force_ask=False, world=world, seed=41)
    st2 = _run_arm("C2", force_ask=False, world=world, seed=41, c_int=10.0)
    assert st2.clarification_count <= st1.clarification_count
    assert st2.clarification_count < st1.clarification_count or st1.clarification_count == 0


def test_contender_L_j_not_all_ones():
    world = _make_world()
    ev = world.events[0]
    mu = np.zeros(6)
    Sigma = np.eye(6)
    score_sys = ev.s0_tilde + ev.phi @ mu
    _, L_j = _compute_contender(
        ev, mu, Sigma, score_sys,
        M=5, n_draws=50,
        rng=np.random.default_rng(0),
    )
    assert not np.allclose(L_j, np.ones(6))


def test_c2_evoi_does_not_depend_on_beta_star():
    import experiment_arms
    original = experiment_arms._array_evoi_for_question

    def wrapper(*args, **kwargs):
        assert "beta_star" not in kwargs
        assert "beta" not in kwargs
        return original(*args, **kwargs)

    experiment_arms._array_evoi_for_question = wrapper
    try:
        world = _make_world()
        _run_arm("C2", force_ask=False, world=world, seed=71)
    finally:
        experiment_arms._array_evoi_for_question = original


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


def test_c3_actually_pairs_and_no_attribute_question():
    world = _make_world()
    st = run_episode(world=world, arm="C3", p_crit=0.0, c_int=0.05,
                     rng=np.random.default_rng(4), evoi_mc_draws=20)
    assert st.pairwise_count >= 1
    assert len(st.question_trace) == 0


def test_c4_stays_prior():
    world = _make_world()
    st = run_episode(world=world, arm="C4", p_crit=0.0, c_int=0.05,
                     rng=np.random.default_rng(4), evoi_mc_draws=20)
    assert np.allclose(st.mu, np.zeros(6))
    assert np.allclose(st.Sigma, np.eye(6))
    assert len(st.evidence_log) == 0


def test_s0_does_not_change_user_true_best():
    from synthetic_user import generate_trip_world
    world = generate_trip_world(
        n_events=2,
        n_candidates=5,
        beta_star=None,
        residual_multiplier=0.0,
        rng=np.random.default_rng(31),
        world_seed=31,
    )
    original_true = [ev.true_best_index for ev in world.events]
    modified = copy.deepcopy(world)
    for ev in modified.events:
        ev.s0_tilde = np.full_like(ev.s0_tilde, 100.0)
    modified_true = [ev.true_best_index for ev in modified.events]
    assert original_true == modified_true


def test_episode_true_best_ignores_s0():
    world = _make_world()
    modified = copy.deepcopy(world)
    for ev in modified.events:
        ev.s0_tilde = -np.arange(ev.s0_tilde.size, dtype=float)

    st_orig = _run_arm("C0", world=world, seed=0)
    st_mod = _run_arm("C0", world=modified, seed=0)

    orig_true = [p["true_best"] for p in st_orig.posterior_trace]
    mod_true = [p["true_best"] for p in st_mod.posterior_trace]
    assert orig_true == mod_true

    orig_proposal = [p["proposal"] for p in st_orig.posterior_trace]
    mod_proposal = [p["proposal"] for p in st_mod.posterior_trace]
    assert any(o != m for o, m in zip(orig_proposal, mod_proposal))


def test_posterior_trace_one_entry_per_event():
    world = _make_world()
    for arm in ["C0", "C1", "C2", "C3", "C4"]:
        st = _run_arm(arm, world=world, seed=100)
        assert len(st.posterior_trace) == len(world.events)
        for i, snap in enumerate(st.posterior_trace):
            assert snap["event_idx"] == i
        assert np.allclose(st.posterior_trace[-1]["mu"], st.mu)
        assert np.allclose(st.posterior_trace[-1]["Sigma_diag"], np.diag(st.Sigma))
    st4 = _run_arm("C4", world=world, seed=101)
    for snap in st4.posterior_trace:
        assert np.allclose(snap["mu"], np.zeros(6))
        assert np.allclose(snap["Sigma_diag"], np.ones(6))
