"""Phase C2 focused tests: EVOI."""

import numpy as np
import pytest

from decision_engine import compute_evoi_for_questions, RankedShop, compute_trip_frozen_scaling, freeze_phase_b_turn_context
from evidence import EvidenceRecord
from likelihood import refit_laplace


def _shop(name, tags):
    from types import SimpleNamespace
    return SimpleNamespace(
        name=name,
        tags=tags,
        flavor_intensity=0.5,
        portion_strictness=0.5,
        review_count=0,
        authority_data=SimpleNamespace(review_count=0, tablelog_medal=""),
        base_wait_minutes=0,
        price_level=2.5,
        default_travel_minutes=15,
    )


def _ranked(shop, score):
    return RankedShop(shop=shop, final_score=float(score))


def _frozen_feature_ctx(shops, ctx):
    means, stds = compute_trip_frozen_scaling(shops, ctx)
    return [float(x) for x in means], [float(x) for x in stds]


def _make_turn_context(ranked, ctx, feature_means, feature_stds):
    return freeze_phase_b_turn_context(
        ranked, ctx, slot_id="lunch", turn_id="t",
        feature_means=feature_means, feature_stds=feature_stds,
    )


def test_p_o_sums_to_one():
    shops = [_shop("A", ["cafe"]), _shop("B", ["ramen"])]
    ctx = {"preferred_tags": ["ramen"]}
    fm, fs = _frozen_feature_ctx(shops, ctx)
    ranked = [_ranked(shops[0], 100), _ranked(shops[1], 99)]
    context = _make_turn_context(ranked, ctx, fm, fs)

    Sigma = np.eye(6)
    mu = np.zeros(6)
    L_j = [1.0] * 6
    x_e = [0.5, 0.5, 0.0, 0.5, 0.0, 0.0]
    questions = [{
        "j_T": 0,
        "j_C": 3,
        "question_options": ["cuisine_match", "travel_min", "other"],
        "score": 1.0,
    }]
    results = compute_evoi_for_questions(
        questions=questions,
        ranked=ranked,
        mu=mu,
        Sigma=Sigma,
        phase_b_context=context,
        x_e=x_e,
        evidences=[],
        c_int=0.05,
        mc_draws=200,
        seed=42,
        ctx=ctx,
    )
    assert len(results) == 1
    p_o = results[0]["p_o"]
    assert sum(p_o) == pytest.approx(1.0, rel=1e-9)


def test_hypothetical_does_not_mutate_real_evidence():
    shops = [_shop("A", ["cafe"]), _shop("B", ["ramen"])]
    ctx = {"preferred_tags": ["ramen"]}
    fm, fs = _frozen_feature_ctx(shops, ctx)
    ranked = [_ranked(shops[0], 100), _ranked(shops[1], 99)]
    context = _make_turn_context(ranked, ctx, fm, fs)

    Sigma = np.eye(6)
    mu = np.zeros(6)
    L_j = [1.0] * 6
    x_e = [0.5, 0.5, 0.0, 0.5, 0.0, 0.0]
    questions = [{
        "j_T": 0,
        "j_C": 3,
        "question_options": ["cuisine_match", "travel_min", "other"],
        "score": 1.0,
    }]
    # build real evidence log with one replacement
    real_ev = [
        EvidenceRecord(
            evidence_id="replace1",
            thread_id="t",
            ts="2024-01-01T00:00:00",
            event_type="replacement",
            learning=True,
            censored_feasibility=False,
            x_e=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ask_eligible=False,
        )
    ]
    before_len = len(real_ev)
    _ = compute_evoi_for_questions(
        questions=questions,
        ranked=ranked,
        mu=mu,
        Sigma=Sigma,
        phase_b_context=context,
        x_e=x_e,
        evidences=real_ev,
        c_int=0.05,
        mc_draws=50,
        seed=0,
        ctx=ctx,
    )
    assert len(real_ev) == before_len
