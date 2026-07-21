"""Unit tests for ``retrieval_service`` seed filtering."""

from decision_engine import ItinerarySynthesizer
from shop_planning import BookingType, QueueStrategy, ShopProfile

from retrieval_service import (
    DietaryConstraints,
    catalog_stem_for_city,
    filter_shop_profiles_by_dietary_exclusions,
    filter_shop_profiles_by_excluded_shop_names,
    normalized_excluded_shop_names_from_intent,
    retrieve_seed_candidates,
    shop_name_matches_exclusion,
)


def test_catalog_stem_for_city() -> None:
    assert catalog_stem_for_city("台北") == "taipei"
    assert catalog_stem_for_city("京都") == "kyoto"
    assert catalog_stem_for_city("東京") == "tokyo"
    assert catalog_stem_for_city("") == "kyoto"


def test_no_beef_excludes_gyukatsu_seed() -> None:
    excl = ItinerarySynthesizer.excluded_tags_for_dietary_key("no_beef")
    cand = retrieve_seed_candidates(
        city="京都",
        dietary_constraints=DietaryConstraints(excluded_shop_tags=excl),
        category_tags=[],
        meal_slots=[],
    )
    names = {s.name for s in cand}
    assert "京都勝牛 河原町店" not in names


def test_ingress_filter_drops_shop_with_mapped_tag() -> None:
    veg = ItinerarySynthesizer.union_excluded_tags_from_dietary_keys(["vegetarian"])
    s = ShopProfile(
        name="KatsuDemo",
        close_time="22:00",
        booking_type=BookingType.NONE,
        queue_strategy=QueueStrategy.PHYSICAL_LINE,
        last_call_offset=30,
        is_cash_only=False,
        sns_handle="demo",
        tags=["beef_cutlet", "set_meal"],
    )
    assert filter_shop_profiles_by_dietary_exclusions([s], veg) == []


def test_meal_slot_filter_falls_back_when_too_strict() -> None:
    """If category+slot would empty the pool, service relaxes constraints (keeps dietary)."""
    cand = retrieve_seed_candidates(
        city="京都",
        dietary_constraints=DietaryConstraints(),
        category_tags=["nonexistent_tag_xyz"],
        meal_slots=["breakfast"],
    )
    assert isinstance(cand, list)
    assert len(cand) >= 1


def test_shop_name_matches_exclusion_compact() -> None:
    assert shop_name_matches_exclusion("茶寮 都路里 祇園本店", "茶寮都路里")
    assert not shop_name_matches_exclusion("其他拉麵", "茶寮都路里")


def test_filter_excluded_shop_names_drops_profiles() -> None:
    keep = ShopProfile(
        name="Other Cafe",
        close_time="22:00",
        booking_type=BookingType.NONE,
        queue_strategy=QueueStrategy.PHYSICAL_LINE,
        last_call_offset=30,
        is_cash_only=False,
        sns_handle="demo",
        tags=["cafe"],
    )
    drop = ShopProfile(
        name="茶寮 都路里 祇園本店",
        close_time="22:00",
        booking_type=BookingType.NONE,
        queue_strategy=QueueStrategy.PHYSICAL_LINE,
        last_call_offset=30,
        is_cash_only=False,
        sns_handle="demo2",
        tags=["dessert"],
    )
    out = filter_shop_profiles_by_excluded_shop_names([keep, drop], ["茶寮都路里"])
    assert [s.name for s in out] == ["Other Cafe"]


def test_normalized_excluded_shop_names_from_intent() -> None:
    assert normalized_excluded_shop_names_from_intent({"excluded_shops": [" X ", "", " X "]}) == ("X",)


def test_retrieve_seed_excludes_matching_catalog_name() -> None:
    kyoto_candidates = retrieve_seed_candidates(
        city="京都",
        dietary_constraints=DietaryConstraints(excluded_shop_names=("燃えよ麺助",)),
        category_tags=[],
        meal_slots=[],
    )
    assert all(s.name != "燃えよ麺助" for s in kyoto_candidates)
