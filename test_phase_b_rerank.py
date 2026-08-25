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


def test_mu_zero_returns_same_order():
    items = [_ranked("A", 100), _ranked("B", 80), _ranked("C", 60)]
    mu = [0.0] * 6
    out = rerank_by_posterior(items, mu, {})
    assert out == items


def test_mu_zero_with_numpy_array():
    items = [_ranked("A", 50), _ranked("B", 40)]
    mu = np.zeros(6)
    out = rerank_by_posterior(items, mu, {})
    assert out == items


def test_nonzero_mu_can_rerank():
    items = [
        _ranked("A", 90),
        _ranked("B", 85),
        _ranked("C", 80),
    ]
    # Make B the only shop matching 'ramen', so cuisine_match >0
    items[1].shop.tags = ["ramen"]
    mu = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ctx = {"preferred_tags": ["ramen"]}
    out = rerank_by_posterior(items, mu, ctx)
    assert out[0].shop.name == "B"
