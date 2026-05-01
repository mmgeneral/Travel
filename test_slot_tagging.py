"""Slot-specific tag OR-groups vs global tags (tea/dinner constraints)."""

from decision_engine import ItinerarySynthesizer
from shop_planning import BookingType, QueueStrategy, ShopProfile


def _shop(**kwargs: object) -> ShopProfile:
    defaults: dict[str, object] = {
        "name": "TestShop",
        "close_time": "22:00",
        "booking_type": BookingType.WALK_IN,
        "queue_strategy": QueueStrategy.PHYSICAL_LINE,
        "last_call_offset": 30,
        "is_cash_only": False,
        "sns_handle": "@t",
        "tags": [],
        "occasion_tags": set(),
    }
    defaults.update(kwargs)
    return ShopProfile(**defaults)


def test_shop_matches_slot_tea_cake_ok() -> None:
    s = _shop(tags=["cake"])
    req = {"tea": {"cake", "dessert", "cafe"}}
    assert ItinerarySynthesizer._shop_matches_slot_required_tags(s, "tea", req)


def test_shop_matches_slot_ramen_not_afternoon_tea() -> None:
    s = _shop(tags=["ramen"])
    req = {"tea": {"cake", "dessert", "cafe", "bakery"}}
    assert not ItinerarySynthesizer._shop_matches_slot_required_tags(s, "tea", req)


def test_occasion_tags_count_for_slot() -> None:
    s = _shop(tags=["japanese"], occasion_tags={"dessert"})
    req = {"tea": {"dessert"}}
    assert ItinerarySynthesizer._shop_matches_slot_required_tags(s, "tea", req)


def test_slot_level_required_tags_tea_dessert_query() -> None:
    from agent import _slot_level_required_tags

    tags = _slot_level_required_tags("京都 下午茶 蛋糕 三家店")
    assert "tea" in tags
    assert tags["tea"] & {"cake", "dessert", "cafe"}


def test_slot_level_required_tags_dinner_izakaya() -> None:
    from agent import _slot_level_required_tags

    tags = _slot_level_required_tags("京都 晚餐 居酒屋 啤酒")
    assert "dinner" in tags
    assert tags["dinner"] & {"yakitori", "izakaya", "beer"}


def test_plan_global_tags_cleared_when_slots_cover_intent() -> None:
    """HARD_TAG 全域集合在時段標籤已涵蓋甜點意圖時應清空，避免單一 genre 霸凌整段行程。"""
    from agent import _plan_global_explicit_tags

    assert _plan_global_explicit_tags("京都 下午茶 蛋糕 三家店") == set()


def test_should_damp_preference_for_explicit_dessert() -> None:
    from agent import _should_damp_preference_for_query

    assert _should_damp_preference_for_query("巴斯克 蛋糕 下午茶")


def test_runtime_damping_reduces_preference_bias() -> None:
    from agent import _apply_runtime_weight_damping
    from decision_engine import WeightProfile

    w = WeightProfile(trust_bias=0.4, preference_bias=0.55, logistics_bias=0.05)
    d = _apply_runtime_weight_damping(w)
    assert d.preference_bias < w.preference_bias
    assert d.trust_bias >= w.trust_bias


def test_slot_triggered_queries_tea_and_dinner_first() -> None:
    """Slot-triggered bundles precede raw query so Places pulls dessert / izakaya into pool."""
    from agent import _plan_dynamic_place_queries, _build_shop_catalog

    queries, _uncovered = _plan_dynamic_place_queries(
        "京都 下午茶 蛋糕 晚餐 串燒",
        "京都",
        list(_build_shop_catalog()),
    )
    assert len(queries) >= 3
    q0 = queries[0].lower()
    assert "dessert" in q0 or "cake" in q0 or "patisserie" in q0
    assert any("izakaya" in q.lower() or "yakitori" in q.lower() for q in queries[:3])
