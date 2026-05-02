"""Tests for decision_engine.ItinerarySynthesizer."""

from datetime import datetime, timedelta

import pytest

from decision_engine import (
    GraphBuilder,
    GraphEdge,
    GraphNode,
    SpatioTemporalGraph,
    ItinerarySynthesizer,
    OptimizationMode,
    RankingEngine,
    RankedShop,
    ScoringEngine,
    UserMinefield,
    UserPreference,
)
from shop_planning import (
    AuthorityData,
    BookingType,
    FlavorCategory,
    MockTrafficProvider,
    QueueStrategy,
    ShopProfile,
)


def _minimal_shop(**kwargs: object) -> ShopProfile:
    defaults: dict[str, object] = {
        "name": "TestShop",
        "close_time": "22:00",
        "booking_type": BookingType.NONE,
        "queue_strategy": QueueStrategy.PHYSICAL_LINE,
        "last_call_offset": 30,
        "is_cash_only": False,
        "sns_handle": "test_shop",
        "avg_eat_minutes": 60,
        "occasion_tags": {"main_meal", "lunch"},
        "tags": ["ramen"],
    }
    defaults.update(kwargs)
    return ShopProfile(**defaults)  # type: ignore[arg-type]


def test_calculate_cooldown_by_flavor_category():
    assert ItinerarySynthesizer._calculate_cooldown(
        _minimal_shop(name="H", flavor_category=FlavorCategory.HEAVY)
    ) == 90
    assert ItinerarySynthesizer._calculate_cooldown(
        _minimal_shop(name="L", flavor_category=FlavorCategory.LIGHT)
    ) == 60
    assert ItinerarySynthesizer._calculate_cooldown(
        _minimal_shop(name="R", flavor_category=FlavorCategory.REFRESHING)
    ) == 45
    assert ItinerarySynthesizer._calculate_cooldown(
        _minimal_shop(name="S", flavor_category=FlavorCategory.SWEET)
    ) == 30
    assert ItinerarySynthesizer._calculate_cooldown(None) == 90
    assert (
        ItinerarySynthesizer._calculate_cooldown(
            _minimal_shop(name="H2", flavor_category=FlavorCategory.HEAVY),
            mode=OptimizationMode.TASTE_MAX,
            requested_meal_count=3,
        )
        == 75
    )
    assert (
        ItinerarySynthesizer._calculate_cooldown(
            _minimal_shop(name="H3", flavor_category=FlavorCategory.HEAVY),
            mode=OptimizationMode.BALANCED,
            requested_meal_count=3,
            appetite_light_mode=True,
        )
        == 60
    )
    assert (
        ItinerarySynthesizer._calculate_cooldown(
            _minimal_shop(
                name="SmallHeavy",
                flavor_category=FlavorCategory.HEAVY,
                has_small_portion=True,
            )
        )
        == max(25, int(round(90 * 0.88)))
    )


def test_fame_damped_final_score_underdog():
    chain_michelin = _minimal_shop(
        name="ChainAward",
        tags=["ramen", "大型連鎖"],
        authority_data=AuthorityData(michelin_star=1),
    )
    raw = 80.0
    damped_off = ScoringEngine.fame_damped_final_score(raw, chain_michelin, underdog_mode=False)
    damped_on = ScoringEngine.fame_damped_final_score(raw, chain_michelin, underdog_mode=True)
    assert damped_off == raw
    acc = ScoringEngine.accolade_bonus(chain_michelin)
    noise = float(chain_michelin.marketing_noise_score or 0.0)
    assert noise == 0.8
    assert damped_on == max(0.0, raw - acc * noise)


def test_low_key_bonus_qualifies_mid_review_no_medal_high_rating():
    humble = _minimal_shop(
        name="HumbleGem",
        google_rating=4.35,
        authority_data=AuthorityData(review_count=180, tablelog_medal="", michelin_star=0),
    )
    assert RankingEngine._qualifies_low_key_bonus(humble)

    crowded = _minimal_shop(
        name="Crowded",
        google_rating=4.35,
        authority_data=AuthorityData(review_count=540, tablelog_medal="", michelin_star=0),
    )
    assert not RankingEngine._qualifies_low_key_bonus(crowded)

    awarded = _minimal_shop(
        name="Awarded",
        google_rating=4.35,
        authority_data=AuthorityData(review_count=180, tablelog_medal="百名店 / 銅賞", michelin_star=0),
    )
    assert not RankingEngine._qualifies_low_key_bonus(awarded)


def test_synthesize_delays_meal_to_open_time():
    traffic = MockTrafficProvider()
    pref = UserPreference()
    start = datetime(2026, 4, 27, 10, 0, 0)
    shop = _minimal_shop(name="LateOpen", open_time="12:00", close_time="22:00")
    ranked = [RankedShop(shop=shop, final_score=90.0)]
    result = ItinerarySynthesizer.synthesize(
        ranked,
        traffic,
        pref,
        start_time=start,
        meal_slots=["lunch"],
        mode=OptimizationMode.TASTE_MAX,
    )
    meal_nodes = [n for n in result.nodes if "Transit to" not in n.title and "BackupNode" not in n.title]
    assert not meal_nodes
    assert any("OPERATING_BOUNDARY_SKIP" in w and "open_time=12:00" in w for w in result.warnings)


def test_synthesize_skips_when_open_alignment_breaks_slot_window():
    traffic = MockTrafficProvider()
    pref = UserPreference()
    start = datetime(2026, 4, 27, 7, 0, 0)
    # Breakfast slot requires breakfast (or ramen global intent bypass); tags keep the
    # scenario about open-time feasibility, not breakfast gating.
    early_shop = _minimal_shop(
        name="OpensAt11",
        open_time="11:00",
        close_time="22:00",
        tags=["ramen", "breakfast"],
        occasion_tags={"breakfast"},
    )
    fallback = _minimal_shop(
        name="EarlyBird",
        open_time="07:00",
        close_time="21:00",
        tags=["ramen", "breakfast"],
        occasion_tags={"breakfast"},
    )
    ranked = [
        RankedShop(shop=early_shop, final_score=95.0),
        RankedShop(shop=fallback, final_score=80.0),
    ]
    result = ItinerarySynthesizer.synthesize(
        ranked,
        traffic,
        pref,
        start_time=start,
        meal_slots=["breakfast"],
        mode=OptimizationMode.TASTE_MAX,
    )
    assert any("OPERATING_BOUNDARY_SKIP" in w and "OpensAt11" in w for w in result.warnings)
    meal_titles = " ".join(n.title for n in result.nodes if "Transit" not in n.title and "Backup" not in n.title)
    assert "EarlyBird" in meal_titles
    assert "OpensAt11" not in meal_titles


def test_slot_semantic_bonus_is_soft_preference():
    breakfast_shop = RankedShop(
        shop=_minimal_shop(name="BreakfastRamen", tags=["ramen", "breakfast"], occasion_tags={"breakfast"}),
        final_score=100.0,
    )
    generic_shop = RankedShop(
        shop=_minimal_shop(name="GenericRamen", tags=["ramen"], occasion_tags={"main_meal"}),
        final_score=100.0,
    )
    # min(1.35, 1.0 + round10(slot_semantic_match_score)/1000); breakfast slot matrix + tags.
    assert ItinerarySynthesizer._slot_semantic_bonus_multiplier("breakfast", breakfast_shop) == pytest.approx(
        1.072
    )
    # Generic ramen-only still picks up modest "breakfast"+"ramen" matrix overlap via tags.
    assert ItinerarySynthesizer._slot_semantic_bonus_multiplier("breakfast", generic_shop) == pytest.approx(1.012)
    assert (
        ItinerarySynthesizer._slot_semantic_bonus_multiplier("breakfast", breakfast_shop)
        > ItinerarySynthesizer._slot_semantic_bonus_multiplier("breakfast", generic_shop)
    )


def test_generate_top_picks_hard_must_have_tag_filter():
    pref = UserPreference()
    minefield = UserMinefield()
    ramen = _minimal_shop(name="RamenA", tags=["ramen"], occasion_tags={"lunch"})
    non_ramen = _minimal_shop(name="CafeB", tags=["cafe"], occasion_tags={"tea"})
    ranked, rejected = RankingEngine.generate_top_picks(
        shops=[ramen, non_ramen],
        preference=pref,
        minefield=minefield,
        must_have_tags=["ramen"],
    )
    assert any(x.shop.name == "RamenA" for x in ranked)
    assert any(x.shop_name == "CafeB" and "TAG_MISMATCH" in x.reason for x in rejected)


def test_last_call_does_not_roll_to_next_day():
    traffic = MockTrafficProvider()
    pref = UserPreference()
    # close=22:00, last_call_offset=30 -> last_call should stay at same-day 21:30.
    late_shop = _minimal_shop(
        name="LateClosed",
        open_time="07:00",
        close_time="22:00",
        last_call_offset=30,
        avg_eat_minutes=45,
    )
    result = ItinerarySynthesizer.synthesize(
        [RankedShop(shop=late_shop, final_score=90.0)],
        traffic,
        pref,
        start_time=datetime(2026, 4, 27, 23, 30, 0),
        meal_slots=[None],
        mode=OptimizationMode.TASTE_MAX,
    )
    assert not result.nodes
    assert any("CONSTRAINT_LAST_CALL_EXCEEDED" in w and "LateClosed" in w for w in result.warnings)


def test_early_bird_shop_is_prioritized_when_feasible():
    traffic = MockTrafficProvider()
    pref = UserPreference()
    start = datetime(2026, 4, 27, 7, 0, 0)
    # last_call=10:40 (close 11:00, offset 20) => early-bird shop
    early_bird = _minimal_shop(
        name="AsaRamen",
        open_time="06:00",
        close_time="11:00",
        last_call_offset=20,
        avg_eat_minutes=35,
        tags=["ramen", "breakfast"],
        occasion_tags={"breakfast", "ramen"},
    )
    regular = _minimal_shop(
        name="RegularLunch",
        open_time="11:00",
        close_time="21:00",
        last_call_offset=30,
        avg_eat_minutes=45,
        tags=["ramen", "lunch"],
        occasion_tags={"lunch", "ramen"},
    )
    ranked = [
        RankedShop(shop=regular, final_score=99.0),
        RankedShop(shop=early_bird, final_score=85.0),
    ]
    result = ItinerarySynthesizer.synthesize(
        ranked,
        traffic,
        pref,
        start_time=start,
        meal_slots=["breakfast"],
        mode=OptimizationMode.BALANCED,
    )
    meal_nodes = [n for n in result.nodes if "·" in n.title]
    assert meal_nodes
    assert "AsaRamen" in meal_nodes[0].title


def test_lunch_slot_does_not_push_early_bird_to_noon():
    traffic = MockTrafficProvider()
    pref = UserPreference()
    start = datetime(2026, 4, 27, 7, 0, 0)
    early_bird = _minimal_shop(
        name="MorningRamen",
        open_time="06:00",
        close_time="11:00",
        last_call_offset=20,
        avg_eat_minutes=30,
        tags=["ramen", "breakfast"],
        occasion_tags={"breakfast", "ramen"},
    )
    ranked = [RankedShop(shop=early_bird, final_score=88.0)]
    result = ItinerarySynthesizer.synthesize(
        ranked,
        traffic,
        pref,
        start_time=start,
        meal_slots=["lunch"],  # auto-partition might produce lunch first
        mode=OptimizationMode.BALANCED,
    )
    meal_nodes = [n for n in result.nodes if "·" in n.title]
    assert meal_nodes
    assert meal_nodes[0].start_at.hour < 12


def test_breakfast_slot_requires_breakfast_tag_and_open_window():
    traffic = MockTrafficProvider()
    pref = UserPreference()
    start = datetime(2026, 4, 27, 7, 0, 0)
    non_breakfast = _minimal_shop(
        name="NoBreakfastTag",
        open_time="06:00",
        close_time="20:00",
        tags=["ramen"],
        occasion_tags={"lunch", "ramen"},
    )
    breakfast_ok = _minimal_shop(
        name="BreakfastRamen",
        open_time="06:00",
        close_time="11:00",
        tags=["ramen", "breakfast"],
        occasion_tags={"breakfast", "ramen"},
    )
    ranked = [
        RankedShop(shop=non_breakfast, final_score=98.0),
        RankedShop(shop=breakfast_ok, final_score=82.0),
    ]
    result = ItinerarySynthesizer.synthesize(
        ranked,
        traffic,
        pref,
        start_time=start,
        meal_slots=["breakfast"],
        mode=OptimizationMode.BALANCED,
    )
    meal_nodes = [n for n in result.nodes if "·" in n.title]
    assert meal_nodes
    assert "BreakfastRamen" in meal_nodes[0].title
    assert any("BREAKFAST_TAG_REQUIRED_SKIP NoBreakfastTag" in w for w in result.warnings)


def test_late_night_slot_prioritizes_izakaya_anchor_tag():
    traffic = MockTrafficProvider()
    pref = UserPreference()
    start = datetime(2026, 4, 27, 22, 0, 0)
    plain = _minimal_shop(
        name="PlainNight",
        open_time="18:00",
        close_time="23:59",
        tags=["snack"],
        occasion_tags={"late_night"},
    )
    izakaya = _minimal_shop(
        name="IzakayaNight",
        open_time="18:00",
        close_time="23:59",
        tags=["izakaya", "late_night"],
        occasion_tags={"late_night", "social"},
    )
    # Graph-optimal picks the higher final_score unless tied; skew taste so anchor-friendly
    # izakaya aligns with DAG preference and slot ordering boost.
    ranked = [
        RankedShop(shop=plain, final_score=93.0),
        RankedShop(shop=izakaya, final_score=95.0),
    ]
    result = ItinerarySynthesizer.synthesize(
        ranked,
        traffic,
        pref,
        start_time=start,
        meal_slots=["late_night"],
        mode=OptimizationMode.BALANCED,
    )
    meal_nodes = [n for n in result.nodes if "·" in n.title]
    assert meal_nodes
    assert "IzakayaNight" in meal_nodes[0].title


def test_explicit_required_tags_becomes_hard_filter_in_synthesis():
    traffic = MockTrafficProvider()
    pref = UserPreference()
    start = datetime(2026, 4, 27, 19, 0, 0)
    ramen = _minimal_shop(name="RamenOnly", open_time="11:00", close_time="23:00", tags=["ramen"])
    izakaya = _minimal_shop(name="IzakayaOnly", open_time="18:00", close_time="23:59", tags=["izakaya"])
    ranked = [
        RankedShop(shop=ramen, final_score=99.0),
        RankedShop(shop=izakaya, final_score=80.0),
    ]
    result = ItinerarySynthesizer.synthesize(
        ranked,
        traffic,
        pref,
        start_time=start,
        meal_slots=["dinner"],
        mode=OptimizationMode.BALANCED,
        explicit_required_tags={"izakaya"},
    )
    meal_nodes = [n for n in result.nodes if "·" in n.title]
    assert meal_nodes
    assert "IzakayaOnly" in meal_nodes[0].title
    assert not any("RamenOnly" in n.title for n in meal_nodes)
    # Explicit tags constrain `find_optimal_path`; the favored shop is izakaya, so ramen
    # (higher ranked score but tag-incompatible with must-have) never reaches HARD_FILTER staging.
    assert any(
        "GRAPH_OPTIMAL_PATH" in w and "IzakayaOnly" in w for w in result.warnings
    )


def test_scarcity_bonus_prioritizes_short_window_shop():
    traffic = MockTrafficProvider()
    pref = UserPreference()
    start = datetime(2026, 4, 27, 7, 0, 0)
    # 4-hour morning window (scarce)
    scarce = _minimal_shop(
        name="AsaLimited",
        open_time="06:00",
        close_time="10:00",
        tags=["ramen", "breakfast"],
        occasion_tags={"breakfast"},
    )
    # Near all-day window
    all_day = _minimal_shop(
        name="AllDayRamen",
        open_time="06:00",
        close_time="23:00",
        tags=["ramen", "breakfast"],
        occasion_tags={"breakfast"},
    )
    ranked = [
        RankedShop(shop=all_day, final_score=95.0),
        RankedShop(shop=scarce, final_score=85.0),
    ]
    assert ItinerarySynthesizer._scarcity_bonus(start, scarce) > ItinerarySynthesizer._scarcity_bonus(
        start,
        all_day,
    )
    result = ItinerarySynthesizer.synthesize(
        ranked,
        traffic,
        pref,
        start_time=start,
        meal_slots=["breakfast"],
        mode=OptimizationMode.BALANCED,
    )
    meal_nodes = [n for n in result.nodes if "·" in n.title]
    assert meal_nodes
    # DAG optimal path favors higher final_score when building preferred_shop_by_slot;
    # scarcity is a ranking tie-layer after graph preference.
    assert "AllDayRamen" in meal_nodes[0].title


def test_graph_builder_creates_compatible_edge():
    traffic = MockTrafficProvider()
    start = datetime(2026, 4, 27, 7, 0, 0)
    breakfast_shop = _minimal_shop(
        name="B1",
        open_time="06:00",
        close_time="11:00",
        avg_eat_minutes=30,
        tags=["ramen", "breakfast"],
        occasion_tags={"breakfast"},
        flavor_category=FlavorCategory.LIGHT,
    )
    lunch_shop = _minimal_shop(
        name="L1",
        open_time="10:30",
        close_time="21:00",
        avg_eat_minutes=45,
        tags=["ramen", "lunch"],
        occasion_tags={"lunch"},
        flavor_category=FlavorCategory.LIGHT,
    )
    ranked = [
        RankedShop(shop=breakfast_shop, final_score=80.0),
        RankedShop(shop=lunch_shop, final_score=90.0),
    ]
    graph = GraphBuilder.build_graph(
        ranked=ranked,
        traffic=traffic,
        start_time=start,
        meal_slots=["breakfast", "lunch"],
        mode=OptimizationMode.BALANCED,
        requested_meal_count=2,
    )
    assert any(n.shop_name == "B1" and n.slot_index == 0 for n in graph.nodes)
    assert any(n.shop_name == "L1" and n.slot_index == 1 for n in graph.nodes)
    lunch_profile = lunch_shop
    queue_risk = min(1.0, max(0.0, float(lunch_profile.base_wait_minutes) / 45.0))
    # Default mock slack swallows route buffer ⇒ travel_buffer_gap=0 here; penalty is queue-only.
    travel_buffer_gap = 0.0
    expected_edge_weight = ScoringEngine.risk_adjusted_score(
        float(ranked[1].final_score),
        queue_risk=queue_risk,
        travel_buffer_gap=travel_buffer_gap,
    )
    b_ids = {n.node_id for n in graph.nodes if n.shop_name == "B1"}
    l_ids = {n.node_id for n in graph.nodes if n.shop_name == "L1"}
    b1_to_l1 = [e for e in graph.edges if e.from_node_id in b_ids and e.to_node_id in l_ids]
    assert b1_to_l1
    assert any(e.weight == pytest.approx(expected_edge_weight) for e in b1_to_l1)


def test_find_optimal_path_respects_must_have_tag_coverage():
    base = datetime(2026, 4, 27, 7, 0, 0)
    g = SpatioTemporalGraph(
        nodes=[
            GraphNode("A0", "AllDayA", 0, base, base + timedelta(minutes=40), 0.0, tags=("all_day",)),
            GraphNode("R1", "Ramen1", 1, base + timedelta(hours=2), base + timedelta(hours=3), 0.0, tags=("ramen",)),
            GraphNode("I1", "Izakaya1", 1, base + timedelta(hours=2), base + timedelta(hours=3), 0.0, tags=("izakaya",)),
        ],
        edges=[
            GraphEdge("A0", "R1", 95.0),
            GraphEdge("A0", "I1", 80.0),
        ],
    )
    path = ItinerarySynthesizer.find_optimal_path(g, required_length=2, must_have_tags={"izakaya"})
    assert path
    assert path[-1].shop_name == "Izakaya1"


def test_find_optimal_path_degrades_to_longest_feasible_length():
    base = datetime(2026, 4, 27, 7, 0, 0)
    g = SpatioTemporalGraph(
        nodes=[
            GraphNode("A0", "A", 0, base, base + timedelta(minutes=30), 0.0, tags=("ramen",)),
            GraphNode("B1", "B", 1, base + timedelta(hours=2), base + timedelta(hours=3), 0.0, tags=("ramen",)),
        ],
        edges=[
            GraphEdge("A0", "B1", 50.0),
        ],
    )
    # required_length=3 impossible; solver should return best length=2 path.
    path = ItinerarySynthesizer.find_optimal_path(g, required_length=3, must_have_tags=set())
    assert len(path) == 2
    assert [n.shop_name for n in path] == ["A", "B"]
