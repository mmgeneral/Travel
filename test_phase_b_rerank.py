"""Phase B2 tests: planner score = S0 + muᵀφ, with real frozen feature scaling."""
import numpy as np
import pytest
from types import SimpleNamespace

from decision_engine import (
    RankedShop,
    compute_trip_frozen_scaling,
    freeze_phase_b_turn_context,
    phase_b_rerank,
)


def _shop(name, tags=None, review_count=0, base_wait=0,
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


def _make_turn_context(ranked, ctx, feature_means, feature_stds):
    return freeze_phase_b_turn_context(
        ranked, ctx, slot_id="lunch", turn_id="t",
        feature_means=feature_means, feature_stds=feature_stds,
    )


def test_mu_zero_returns_same_order_and_exact_scores():
    shops = [
        _shop("A", ["cafe"]),
        _shop("B", ["ramen"]),
        _shop("C", ["sushi"]),
    ]
    ctx = {"preferred_tags": ["ramen"]}
    fm, fs = _frozen_feature_ctx(shops, ctx)
    ranked = [_ranked(shops[i], 100 - i) for i in range(3)]
    context = _make_turn_context(ranked, ctx, fm, fs)
    mu = [0.0] * 6
    out = phase_b_rerank(ranked, mu, context)
    assert [r.shop.name for r in out] == ["A", "B", "C"]
    for r, orig in zip(out, ranked):
        assert r.final_score == pytest.approx(orig.final_score)


def test_mu_zero_with_numpy_array():
    shops = [_shop("A", ["cafe"]), _shop("B", ["ramen"])]
    ctx = {"preferred_tags": ["ramen"]}
    fm, fs = _frozen_feature_ctx(shops, ctx)
    ranked = [_ranked(shops[0], 50), _ranked(shops[1], 40)]
    context = _make_turn_context(ranked, ctx, fm, fs)
    mu = np.zeros(6)
    out = phase_b_rerank(ranked, mu, context)
    assert [r.shop.name for r in out] == ["A", "B"]
    assert out[0].final_score == pytest.approx(50.0)
    assert out[1].final_score == pytest.approx(40.0)


def test_nonzero_mu_can_rerank_using_standardized_phi():
    shops = [
        _shop("A", ["cafe"]),
        _shop("B", ["ramen"]),
        _shop("C", ["sushi"]),
    ]
    ctx = {"preferred_tags": ["ramen"]}
    fm, fs = _frozen_feature_ctx(shops, ctx)
    ranked = [_ranked(shops[0], 90), _ranked(shops[1], 85), _ranked(shops[2], 80)]
    context = _make_turn_context(ranked, ctx, fm, fs)
    mu = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    out = phase_b_rerank(ranked, mu, context)
    assert out[0].shop.name == "B"
