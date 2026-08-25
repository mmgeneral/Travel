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


def _phase_b_context_for_ranked(ranked):
    raw = {r.shop.name: r.final_score for r in ranked}
    vals = list(raw.values())
    mean = float(np.mean(vals))
    std = float(np.std(vals))
    scale = std if std > 1e-12 else 1.0
    means = np.zeros(6)
    stds = np.ones(6)
    return {
        "candidate_names": [r.shop.name for r in ranked],
        "s0_by_candidate": raw,
        "s0_mean": mean,
        "s0_std": std,
        "s0_scale": scale,
        "feature_means": means.tolist(),
        "feature_stds": stds.tolist(),
    }


def test_contender_approx_cand_when_sigma_close_to_identity():
    ranked = _ranked_shops([
        ("A", 100.0),
        ("B", 99.8),
        ("C", 99.6),
        ("D", 99.4),
        ("E", 99.2),
    ])
    # give each shop a distinct simple feature to make φ differences non-zero
    for i, r in enumerate(ranked):
        r.shop.flavor_intensity = 0.1 + 0.1 * i
    mu = [0.0] * 6
    Sigma = np.eye(6)
    ctx_b = _phase_b_context_for_ranked(ranked)
    contender, meta = contender_set(ranked, mu, Sigma, {}, ctx_b)
    # Σ ≈ I 時 margin 較寬鬆，應保留絕大多數候選
    assert meta["size"] >= len(ranked) - 1


def test_contender_shrinks_when_sigma_small():
    ranked = _ranked_shops([
        ("A", 100.0),
        ("B", 55.0),
        ("C", 52.0),
        ("D", 50.0),
    ])
    for i, r in enumerate(ranked):
        r.shop.flavor_intensity = 0.1 + 0.1 * i
    mu = [0.0] * 6
    ctx_b = _phase_b_context_for_ranked(ranked)
    Sigma = 0.001 * np.eye(6)
    contender, meta = contender_set(ranked, mu, Sigma, {}, ctx_b)
    size_small = meta["size"]
    assert size_small < len(ranked)

    Sigma_identity = np.eye(6)
    _, meta_identity = contender_set(ranked, mu, Sigma_identity, {}, ctx_b)
    print("contender_size_identity =", meta_identity["size"])
    print("contender_size_small_sigma =", size_small)
    assert meta_identity["size"] > size_small


def test_contender_short_circuit_when_singleton():
    ranked = _ranked_shops([
        ("A", 1000.0),
        ("B", 10.0),
    ])
    for i, r in enumerate(ranked):
        r.shop.flavor_intensity = 0.1 + 0.1 * i
    mu = [0.0] * 6
    ctx_b = _phase_b_context_for_ranked(ranked)
    Sigma = 1e-8 * np.eye(6)
    contender, meta = contender_set(ranked, mu, Sigma, {}, ctx_b)
    assert meta["size"] == 1
    assert meta["mc_calls"] == 0
    assert meta["gate_short_circuit"] is True
    assert contender[0].shop.name == "A"
    # fake MC should not be called
    mc_calls = 0
    def fake_mc():
        nonlocal mc_calls
        mc_calls += 1
    if not meta["gate_short_circuit"]:
        fake_mc()
    assert mc_calls == 0


def test_L_j_provided():
    ranked = _ranked_shops([
        ("A", 100.0),
        ("B", 90.0),
        ("C", 80.0),
    ])
    for i, r in enumerate(ranked):
        r.shop.flavor_intensity = 0.1 + 0.1 * i
    mu = [0.0] * 6
    ctx_b = _phase_b_context_for_ranked(ranked)
    Sigma = np.eye(6)
    _, meta = contender_set(ranked, mu, Sigma, {}, ctx_b)
    assert "L_j" in meta
    assert len(meta["L_j"]) == 6
