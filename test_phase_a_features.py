"""Phase A1 feature tests."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from preference_features import (
    FEATURE_NAMES,
    FEATURE_NAME_TO_INDEX,
    TASTE_INDICES,
    CONTEXT_INDICES,
)
from decision_engine import phi, compute_trip_frozen_scaling, z_score


def _shop(name, **attrs):
    base = {
        "name": name,
        "tags": [],
        "flavor_intensity": 0.5,
        "portion_strictness": 0.5,
        "review_count": None,
        "authority_data": SimpleNamespace(review_count=0, tablelog_medal=""),
        "base_wait_minutes": 0,
        "price_level": None,
        "default_travel_minutes": 15,
    }
    base.update(attrs)
    return SimpleNamespace(**base)


def test_registry():
    assert FEATURE_NAMES == [
        "cuisine_match",
        "fame_touristy",
        "heaviness",
        "travel_min",
        "price_level",
        "queue_wait",
    ]
    assert FEATURE_NAME_TO_INDEX == {n: i for i, n in enumerate(FEATURE_NAMES)}
    assert TASTE_INDICES == [0, 1, 2]
    assert CONTEXT_INDICES == [3, 4, 5]


def test_fame_uses_authority_review_count():
    zero_review_shop = _shop(
        "Zero",
        tags=["ramen"],
        authority_data=SimpleNamespace(review_count=0, tablelog_medal=""),
    )
    high_review_shop = _shop(
        "High",
        tags=["ramen"],
        authority_data=SimpleNamespace(review_count=150, tablelog_medal=""),
    )
    low = phi(zero_review_shop, {})[1]
    high = phi(high_review_shop, {})[1]
    assert high > low
    assert high == pytest.approx(0.5)


def test_travel_min_candidate_specific():
    item_a = _shop("A")
    item_b = _shop("B")
    ctx = {"travel_minutes_map": {"A": 5.0, "B": 20.0}}
    assert phi(item_a, ctx)[3] == pytest.approx(5.0)
    assert phi(item_b, ctx)[3] == pytest.approx(20.0)


def test_missing_price_produces_no_fake_variance():
    a = _shop("A")
    b = _shop("B")
    fa = phi(a, {})
    fb = phi(b, {})
    assert fa[4] == fb[4] == pytest.approx(2.5)


def test_frozen_standardization_reproducible():
    items = [_shop("A", tags=["ramen"]), _shop("B", tags=["cafe"]), _shop("C", tags=["sushi"])]
    means, stds = compute_trip_frozen_scaling(items, {})
    z1 = z_score(items[0], {}, (means, stds))
    assert len(z1) == 6
    assert np.all(np.isfinite(z1))
    z2 = z_score(items[0], {}, (means, stds))
    assert np.allclose(z1, z2)
    z_same = z_score(_shop("D"), {}, (means, stds))
    assert np.all(np.isfinite(z_same))
