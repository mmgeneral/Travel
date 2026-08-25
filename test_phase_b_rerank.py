"""Phase B2 tests: planner score = S0 + muᵀφ, and μ=0 fallback."""
import numpy as np
from types import SimpleNamespace

from decision_engine import RankedShop, rerank_by_posterior


def _shop(name: str, tags: list[str] | None = None,
          review_count: int = 0, flavor: float = 0.5,
          base_wait: int = 0, price: float = 2.5) -> SimpleNamespace:
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


def _ranked(name: str, score: float) -> RankedShop:
    return RankedShop(shop=_shop(name), final_score=score)


def _phase_b_context(items):
    raw = {r.shop.name: r.final_score for r in items}
    vals = list(raw.values())
    mean = float(np.mean(vals))
    std = float(np.std(vals))
    scale = std if std > 1e-12 else 1.0
    means = np.zeros(6)
    stds = np.ones(6)
    return {
        "candidate_names": [r.shop.name for r in items],
        "s0_by_candidate": raw,
        "s0_mean": mean,
        "s0_std": std,
        "s0_scale": scale,
        "feature_means": means.tolist(),
        "feature_stds": stds.tolist(),
    }


def test_mu_zero_returns_same_order():
    items = [_ranked("A", 100), _ranked("B", 80), _ranked("C", 60)]
    mu = [0.0] * 6
    ctx = _phase_b_context(items)
    out = rerank_by_posterior(items, mu, {}, ctx)
    assert [r.shop.name for r in out] == ["A", "B", "C"]
    # exact S_eff == S0
    for r, orig in zip(out, items):
        assert r.final_score == pytest.approx(orig.final_score)


def test_mu_zero_with_numpy_array():
    items = [_ranked("A", 50), _ranked("B", 40)]
    mu = np.zeros(6)
    ctx = _phase_b_context(items)
    out = rerank_by_posterior(items, mu, {}, ctx)
    assert [r.shop.name for r in out] == ["A", "B"]
    assert out[0].final_score == pytest.approx(50.0)
    assert out[1].final_score == pytest.approx(40.0)


def test_nonzero_mu_can_rerank():
    items = [
        _ranked("A", 90),
        _ranked("B", 85),
        _ranked("C", 80),
    ]
    # Make B the only shop matching 'ramen', so cuisine_match >0
    items[1].shop.tags = ["ramen"]
    items[0].shop.tags = ["cafe"]
    items[2].shop.tags = ["sushi"]
    ctx = {"preferred_tags": ["ramen"]}
    ctx_b = _phase_b_context(items)
    mu = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    out = rerank_by_posterior(items, mu, ctx, ctx_b)
    assert out[0].shop.name == "B"
