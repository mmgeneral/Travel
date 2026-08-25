"""Phase B3 tests: contender set using 2·σ_pred margin filter."""
import numpy as np
import pytest
from types import SimpleNamespace

from decision_engine import (
    RankedShop,
    contender_set,
)


def _shop(name, tags, review_count=0, base_wait=0, flavor=0.5, price=2.5):
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


def _ranked_shops(names_scores):
    ranked = []
    for name, score in names_scores:
        shop = _shop(name, tags=["ramen"])
        ranked.append(RankedShop(shop=shop, final_score=score))
    # 確保最高分在第一個
    ranked.sort(key=lambda r: r.final_score, reverse=True)
    return ranked


def test_contender_approx_cand_when_sigma_close_to_identity():
    ranked = _ranked_shops([
        ("A", 100.0),
        ("B", 99.8),
        ("C", 99.6),
        ("D", 99.4),
        ("E", 99.2),
    ])
    mu = [0.0] * 6
    Sigma = np.eye(6)
    contender, meta = contender_set(ranked, mu, Sigma, {})
    # Σ ≈ I 時 margin 較寬鬆，所有候選都應保留
    assert meta["size"] == len(ranked)


def test_contender_shrinks_when_sigma_small():
    ranked = _ranked_shops([
        ("A", 100.0),
        ("B", 55.0),
        ("C", 52.0),
        ("D", 50.0),
    ])
    mu = [0.0] * 6
    Sigma = 0.001 * np.eye(6)
    contender, meta = contender_set(ranked, mu, Sigma, {})
    size_small = meta["size"]
    assert size_small < len(ranked)

    Sigma_identity = np.eye(6)
    _, meta_identity = contender_set(ranked, mu, Sigma_identity, {})
    print(f"|C̃| with Σ≈I: {meta_identity['size']}")
    print(f"|C̃| with Σ small: {size_small}")
    assert meta_identity["size"] > size_small


def test_contender_short_circuit_when_singleton():
    ranked = _ranked_shops([
        ("A", 1000.0),
        ("B", 10.0),
    ])
    mu = [0.0] * 6
    Sigma = 1e-8 * np.eye(6)
    contender, meta = contender_set(ranked, mu, Sigma, {})
    assert meta["size"] == 1
    assert meta["mc_calls"] == 0
    assert contender[0].shop.name == "A"


def test_L_j_provided():
    ranked = _ranked_shops([
        ("A", 100.0),
        ("B", 90.0),
        ("C", 80.0),
    ])
    mu = [0.0] * 6
    Sigma = np.eye(6)
    _, meta = contender_set(ranked, mu, Sigma, {})
    assert "L_j" in meta
    assert len(meta["L_j"]) == 6
