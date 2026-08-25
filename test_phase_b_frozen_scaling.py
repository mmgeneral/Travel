"""Phase B1 tests: frozen candidate universe scaling."""
from types import SimpleNamespace

import numpy as np

from decision_engine import freeze_candidate_scaling, z_score


def _shop(name: str, tags: list[str], review_count: int = 0,
          base_wait: int = 0, flavor: float = 0.5, price: float = 2.5) -> SimpleNamespace:
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


POOL = [
    _shop("A", ["ramen"], review_count=100, base_wait=10, flavor=0.8),
    _shop("B", ["cafe"], review_count=50, base_wait=20, flavor=0.4),
    _shop("C", ["sushi"], review_count=200, base_wait=5, flavor=0.6),
]


def test_freeze_candidate_scaling_returns_plain_lists() -> None:
    s = freeze_candidate_scaling(POOL, {"preferred_tags": ["ramen"]})
    assert isinstance(s["means"], list)
    assert isinstance(s["stds"], list)
    assert len(s["means"]) == 6
    assert len(s["stds"]) == 6


def test_z_scores_reproducible_same_turn() -> None:
    ctx = {"preferred_tags": ["ramen"]}
    s = freeze_candidate_scaling(POOL, ctx)
    means = np.array(s["means"])
    stds = np.array(s["stds"])
    first = [z_score(item, ctx, (means, stds)).tolist() for item in POOL]
    second = [z_score(item, ctx, (means, stds)).tolist() for item in POOL]
    assert first == second


def test_scaling_not_recomputed_when_ctx_changes() -> None:
    ctx1 = {"preferred_tags": ["ramen"]}
    s = freeze_candidate_scaling(POOL, ctx1)
    before_means = list(s["means"])
    before_stds = list(s["stds"])

    # Hypothetical B_o changes ctx (e.g., travel time)
    ctx_b = {"preferred_tags": ["ramen"], "travel_minutes": 120}
    item = POOL[0]
    _ = z_score(item, ctx_b, (np.array(s["means"]), np.array(s["stds"])))

    assert s["means"] == before_means
    assert s["stds"] == before_stds


def test_frozen_scaling_deterministic() -> None:
    ctx = {"preferred_tags": ["ramen"]}
    s1 = freeze_candidate_scaling(POOL, ctx)
    s2 = freeze_candidate_scaling(POOL, ctx)
    assert s1["means"] == s2["means"]
    assert s1["stds"] == s2["stds"]
