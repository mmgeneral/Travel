"""Phase B3 tests: contender set using 2·σ_pred margin filter."""
import numpy as np
import pytest
from types import SimpleNamespace

from decision_engine import (
    RankedShop,
    compute_trip_frozen_scaling,
    freeze_phase_b_turn_context,
    contender_set,
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


def test_contender_approx_cand_when_sigma_close_to_identity():
    shops = [
        _shop("A", ["cafe"]),
        _shop("B", ["ramen"]),
        _shop("C", ["sushi"]),
        _shop("D", ["beef"]),
        _shop("E", ["chicken"]),
    ]
    ctx = {"preferred_tags": ["ramen"]}
    fm, fs = _frozen_feature_ctx(shops, ctx)
    ranked = [_ranked(shops[i], 100 - i) for i in range(5)]
    context = _make_turn_context(ranked, ctx, fm, fs)
    mu = [0.0] * 6
    c, meta = contender_set(ranked, mu, np.eye(6), {}, context)
    assert meta["size"] >= len(ranked) - 1


def test_contender_shrinks_when_sigma_small():
    shops = [
        _shop("A", ["cafe"]),
        _shop("B", ["ramen"]),
        _shop("C", ["sushi"]),
        _shop("D", ["beef"]),
    ]
    ctx = {"preferred_tags": ["ramen"]}
    fm, fs = _frozen_feature_ctx(shops, ctx)
    ranked = [_ranked(shops[i], 100 - i) for i in range(4)]
    context = _make_turn_context(ranked, ctx, fm, fs)
    mu = [0.0] * 6
    c_small, meta_small = contender_set(ranked, mu, 0.001 * np.eye(6), {}, context)
    c_identity, meta_identity = contender_set(ranked, mu, np.eye(6), {}, context)
    assert meta_identity["size"] > meta_small["size"]


def test_contender_short_circuit_when_singleton():
    shops = [_shop("A", ["cafe"]), _shop("B", ["ramen"])]
    ctx = {"preferred_tags": ["ramen"]}
    fm, fs = _frozen_feature_ctx(shops, ctx)
    ranked = [_ranked(shops[0], 100.0), _ranked(shops[1], 1.0)]
    context = _make_turn_context(ranked, ctx, fm, fs)
    mu = [0.0] * 6
    c, meta = contender_set(ranked, mu, 1e-8 * np.eye(6), {}, context)
    assert meta["size"] == 1
    assert meta["mc_calls"] == 0
    assert meta["gate_short_circuit"] is True
    assert c[0].shop.name == "A"


def test_L_j_provided():
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
    _, meta = contender_set(ranked, mu, np.eye(6), {}, context)
    assert "L_j" in meta
    assert len(meta["L_j"]) == 6


def test_identical_feature_worse_score_excluded():
    shops = [
        _shop("A", ["cafe"]),
        _shop("B", ["cafe"]),
    ]
    ctx = {"preferred_tags": ["ramen"]}
    fm, fs = _frozen_feature_ctx(shops, ctx)
    ranked = [_ranked(shops[0], 100.0), _ranked(shops[1], 90.0)]
    context = _make_turn_context(ranked, ctx, fm, fs)
    mu = [0.0] * 6
    c, meta = contender_set(ranked, mu, np.eye(6), {}, context)
    assert [r.shop.name for r in c] == ["A"]


def test_best_retained():
    shops = [_shop("A", ["cafe"]), _shop("B", ["ramen"])]
    ctx = {"preferred_tags": ["ramen"]}
    fm, fs = _frozen_feature_ctx(shops, ctx)
    ranked = [_ranked(shops[0], 100.0), _ranked(shops[1], 50.0)]
    context = _make_turn_context(ranked, ctx, fm, fs)
    mu = [0.0] * 6
    c, meta = contender_set(ranked, mu, np.eye(6), {}, context)
    assert c[0].shop.name == "A"
