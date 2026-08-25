"""Phase B1 tests: revision-turn frozen S0 + trip-frozen feature scaling."""
from types import SimpleNamespace

import json
import numpy as np
import pytest

from decision_engine import (
    RankedShop,
    compute_trip_frozen_scaling,
    freeze_phase_b_turn_context,
    phase_b_rerank,
    contender_set,
)


def _shop(name: str, tags=None, review_count=0, base_wait=0,
          flavor=0.5, price=2.5):
    tags = tags or []
    return SimpleNamespace(
        name=name,
        tags=tags,
        flavor_intensity=flavor,
        portion_strictness=0.5,
        authority_data=SimpleNamespace(review_count=review_count, tablelog_medal=""),
        base_wait_minutes=base_wait,
        price_level=price,
        default_travel_minutes=15,
    )


def _ranked(shop, score):
    return RankedShop(shop=shop, final_score=float(score))


def _frozen_feature_ctx(shops, ctx):
    means, stds = compute_trip_frozen_scaling(shops, ctx)
    return [float(x) for x in means], [float(x) for x in stds]


def _make_turn_context(ranked, ctx, feature_means, feature_stds, slot_id="lunch", turn_id="t"):
    return freeze_phase_b_turn_context(
        ranked, ctx, slot_id=slot_id, turn_id=turn_id,
        feature_means=feature_means, feature_stds=feature_stds,
    )


def test_trip_feature_scaling_is_frozen_across_revisions():
    shops = [
        _shop("A", ["cafe"], review_count=50),
        _shop("B", ["ramen"], review_count=150),
        _shop("C", ["sushi"], review_count=100),
    ]
    ctx = {"preferred_tags": ["ramen"]}
    fm, fs = _frozen_feature_ctx(shops, ctx)
    ranked1 = [_ranked(shops[1], 90.0)]
    ctx1 = _make_turn_context(ranked1, ctx, fm, fs, slot_id="lunch", turn_id="rev1")
    ranked2 = [_ranked(shops[2], 80.0)]
    ctx2 = _make_turn_context(ranked2, ctx, fm, fs, slot_id="lunch", turn_id="rev2")
    assert ctx1["feature_means"] == fm
    assert ctx2["feature_means"] == fm
    assert ctx1["feature_stds"] == fs
    assert ctx2["feature_stds"] == fs


def test_new_revision_gets_new_s0_scaling_but_same_feature_scaling():
    shops = [_shop("A", ["cafe"]), _shop("B", ["ramen"])]
    ctx = {"preferred_tags": ["ramen"]}
    fm, fs = _frozen_feature_ctx(shops, ctx)
    ranked1 = [_ranked(shops[0], 100.0), _ranked(shops[1], 80.0)]
    ctx1 = _make_turn_context(ranked1, ctx, fm, fs, slot_id="lunch", turn_id="rev1")
    ranked2 = [_ranked(shops[1], 95.0), _ranked(shops[0], 70.0)]
    ctx2 = _make_turn_context(ranked2, ctx, fm, fs, slot_id="lunch", turn_id="rev2")
    assert ctx1["s0_mean"] != ctx2["s0_mean"]
    assert ctx1["s0_scale"] != ctx2["s0_scale"]
    assert ctx1["feature_means"] == ctx2["feature_means"]


def test_phase_b_turn_context_json_serializable():
    shops = [_shop("A", ["cafe"]), _shop("B", ["ramen"])]
    ctx = {"preferred_tags": ["ramen"]}
    fm, fs = _frozen_feature_ctx(shops, ctx)
    ranked = [_ranked(shops[0], 100.0), _ranked(shops[1], 80.0)]
    context = _make_turn_context(ranked, ctx, fm, fs)
    json.dumps(context)


def test_s0_tilde_deterministic_within_turn():
    shops = [_shop("A", ["cafe"]), _shop("B", ["ramen"])]
    ctx = {"preferred_tags": ["ramen"]}
    fm, fs = _frozen_feature_ctx(shops, ctx)
    ranked = [_ranked(shops[0], 100.0), _ranked(shops[1], 80.0)]
    context = _make_turn_context(ranked, ctx, fm, fs)
    assert context["s0_by_candidate"]["A"] == pytest.approx(100.0)
    assert context["s0_by_candidate"]["B"] == pytest.approx(80.0)


def test_mu_zero_exact_score_fallback():
    shops = [
        _shop("A", ["cafe"]),
        _shop("B", ["ramen"]),
        _shop("C", ["sushi"]),
    ]
    ctx = {"preferred_tags": ["ramen"]}
    fm, fs = _frozen_feature_ctx(shops, ctx)
    ranked = [_ranked(shops[i], 100 - i) for i in range(3)]
    context = _make_turn_context(ranked, ctx, fm, fs)
    out = phase_b_rerank(ranked, [0.0] * 6, context)
    assert [r.shop.name for r in out] == ["A", "B", "C"]
    for r, orig in zip(out, ranked):
        assert r.final_score == pytest.approx(orig.final_score)


def test_nonzero_mu_can_rerank_using_standardized_phi():
    shops = [
        _shop("A", ["cafe"]),
        _shop("B", ["ramen"]),
        _shop("C", ["sushi"]),
    ]
    ctx = {"preferred_tags": ["ramen"]}
    fm, fs = _frozen_feature_ctx(shops, ctx)
    ranked = [_ranked(shops[0], 100), _ranked(shops[1], 90), _ranked(shops[2], 80)]
    context = _make_turn_context(ranked, ctx, fm, fs)
    mu = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    out = phase_b_rerank(ranked, mu, context)
    # B has higher cuisine_match than A/C after standardization
    assert out[0].shop.name == "B"


def test_posterior_candidate_can_enter_from_s0_rank6():
    shops = [
        _shop("A", ["cafe"]),
        _shop("B", ["sushi"]),
        _shop("C", ["beef"]),
        _shop("D", ["takoyaki"]),
        _shop("E", ["chicken"]),
        _shop("F", ["ramen"]),
    ]
    ctx = {"preferred_tags": ["ramen"]}
    fm, fs = _frozen_feature_ctx(shops, ctx)
    scores = [100, 95, 90, 85, 80, 60]
    ranked = [_ranked(shops[i], scores[i]) for i in range(6)]
    context = _make_turn_context(ranked, ctx, fm, fs)
    mu = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    out = phase_b_rerank(ranked, mu, context)
    names = [r.shop.name for r in out]
    # F enters top-5
    assert "F" in names[:5]


def test_B3_identity_vs_small_sigma():
    shops = [
        _shop("A", ["cafe"]),
        _shop("B", ["ramen"]),
        _shop("C", ["sushi"]),
        _shop("D", ["beef"]),
        _shop("E", ["chicken"]),
    ]
    ctx = {"preferred_tags": ["ramen"]}
    fm, fs = _frozen_feature_ctx(shops, ctx)
    scores = [100, 99, 98, 97, 96]
    ranked = [_ranked(shops[i], scores[i]) for i in range(5)]
    context = _make_turn_context(ranked, ctx, fm, fs)
    mu = [0.0] * 6
    c_id, _ = contender_set(ranked, mu, np.eye(6), {}, context)
    size_identity = len(c_id)
    c_small, _ = contender_set(ranked, mu, 0.01 * np.eye(6), {}, context)
    size_small = len(c_small)
    assert size_identity > size_small
