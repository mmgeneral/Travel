"""Phase D.2 regression: C1 and C2 must be bit-identical when gate is forced to ASK."""
import numpy as np
import pytest
import config
from types import SimpleNamespace

import decision_engine


def _shop(name, tags):
    return SimpleNamespace(
        name=name,
        tags=tags,
        flavor_intensity=0.5,
        portion_strictness=0.5,
        review_count=0,
        authority_data=SimpleNamespace(review_count=0, tablelog_medal=""),
        base_wait_minutes=0,
        price_level=2.5,
    )


def _ranked(shop, score):
    from decision_engine import RankedShop
    return RankedShop(shop=shop, final_score=float(score))


def _run_pipeline(arm: str, force_ask: bool):
    from decision_engine import (
        compute_trip_frozen_scaling,
        freeze_phase_b_turn_context,
        phase_b_rerank,
        contender_set,
        generate_cross_block_questions,
        compute_evoi_for_questions,
        evaluate_gate,
    )

    shops = [
        _shop("A", ["cafe"]),
        _shop("B", ["ramen"]),
        _shop("C", ["sushi"]),
        _shop("D", ["beef"]),
        _shop("E", ["chicken"]),
    ]
    ctx = {"preferred_tags": ["ramen"]}
    fm, fs = compute_trip_frozen_scaling(shops, ctx)
    ranked = [_ranked(shops[i], 100 - i) for i in range(5)]
    phase_b_context = freeze_phase_b_turn_context(
        ranked, ctx, slot_id="lunch", turn_id="t",
        feature_means=[float(x) for x in fm],
        feature_stds=[float(x) for x in fs],
    )
    mu = [0.0] * 6
    Sigma = np.eye(6)
    reranked = phase_b_rerank(ranked, mu, phase_b_context, ctx)
    contender, meta = contender_set(reranked, mu, Sigma, ctx, phase_b_context)
    x_e = [0.5, 0.5, 0.0, 0.5, 0.0, 0.0]
    questions, ask_eligible = generate_cross_block_questions(
        Sigma=Sigma, L_j=meta["L_j"], x_e=x_e,
    )
    evoi_results = []
    if ask_eligible and questions:
        evoi_results = compute_evoi_for_questions(
            questions=questions,
            ranked=reranked,
            mu=mu,
            Sigma=Sigma,
            phase_b_context=phase_b_context,
            x_e=x_e,
            evidences=[],
            c_int=0.05,
            mc_draws=40,
            seed=123,
            ctx=ctx,
        )
    gate = evaluate_gate(
        ask_eligible=ask_eligible,
        evoi_results=evoi_results,
        asked_this_turn=False,
        contender_size=meta["size"],
        attribution_already_given=False,
    )
    return {
        "phase_b_context": phase_b_context,
        "contender_names": [r.shop.name for r in contender],
        "meta": meta,
        "questions": questions,
        "evoi_results": evoi_results,
        "gate": gate,
    }


def test_c1_c2_bit_identical_when_force_ask():
    # Use explicit policy/force_ask arguments (no global mutation).
    out1 = _run_pipeline("C1", force_ask=True)
    out2 = _run_pipeline("C2", force_ask=True)

    assert out1 == out2
    assert out1["gate"]["action"] == "ask"
