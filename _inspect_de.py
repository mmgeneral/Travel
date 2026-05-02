#!/usr/bin/env python3
from datetime import datetime
from pathlib import Path

from decision_engine import GraphBuilder, ItinerarySynthesizer, OptimizationMode, RankedShop
from shop_planning import BookingType, FlavorCategory, MockTrafficProvider, QueueStrategy, ShopProfile, UserPreference


def _minimal_shop(**kwargs):
    defaults = dict(
        name="TestShop",
        close_time="22:00",
        booking_type=BookingType.NONE,
        queue_strategy=QueueStrategy.PHYSICAL_LINE,
        last_call_offset=30,
        is_cash_only=False,
        sns_handle="test_shop",
        avg_eat_minutes=60,
        occasion_tags={"main_meal", "lunch"},
        tags=["ramen"],
    )
    defaults.update(kwargs)
    return ShopProfile(**defaults)


def main():
    out: list[str] = []
    breakfast_shop = RankedShop(
        shop=_minimal_shop(name="BreakfastRamen", tags=["ramen", "breakfast"], occasion_tags={"breakfast"}),
        final_score=100.0,
    )
    generic_shop = RankedShop(
        shop=_minimal_shop(name="GenericRamen", tags=["ramen"], occasion_tags={"main_meal"}),
        final_score=100.0,
    )
    out.append(
        f"MULT breakfast {ItinerarySynthesizer._slot_semantic_bonus_multiplier('breakfast', breakfast_shop)}"
    )
    out.append(f"MULT generic {ItinerarySynthesizer._slot_semantic_bonus_multiplier('breakfast', generic_shop)}")

    traffic = MockTrafficProvider()
    pref = UserPreference()
    start = datetime(2026, 4, 27, 7, 0, 0)
    early_shop = _minimal_shop(name="OpensAt11", open_time="11:00", close_time="22:00")
    fallback = _minimal_shop(name="EarlyBird", open_time="07:00", close_time="21:00")
    ranked = [
        RankedShop(shop=early_shop, final_score=95.0),
        RankedShop(shop=fallback, final_score=80.0),
    ]
    r2 = ItinerarySynthesizer.synthesize(
        ranked, traffic, pref, start_time=start, meal_slots=["breakfast"], mode=OptimizationMode.TASTE_MAX
    )
    out.append("WARNINGS open_alignment:")
    out.extend([f"  {w}" for w in r2.warnings])
    out.append(f"NODES {[n.title for n in r2.nodes]}")

    ramen = _minimal_shop(name="RamenOnly", open_time="11:00", close_time="23:00", tags=["ramen"])
    izakaya = _minimal_shop(name="IzakayaOnly", open_time="18:00", close_time="23:59", tags=["izakaya"])
    ranked_hf = [
        RankedShop(shop=ramen, final_score=99.0),
        RankedShop(shop=izakaya, final_score=80.0),
    ]
    r_hf = ItinerarySynthesizer.synthesize(
        ranked_hf,
        traffic,
        pref,
        start_time=datetime(2026, 4, 27, 19, 0, 0),
        meal_slots=["dinner"],
        mode=OptimizationMode.BALANCED,
        explicit_required_tags={"izakaya"},
    )
    out.append("WARNINGS hard_filter:")
    out.extend([f"  {w}" for w in r_hf.warnings])
    mids = []
    for n in r_hf.nodes:
        mid = getattr(n, "title", "") or ""
        if "·" in mid:
            mids.append(mid)
    out.append(f"FIRST meal {mids}")

    start5 = datetime(2026, 4, 27, 7, 0, 0)
    scarce = _minimal_shop(
        name="AsaLimited",
        open_time="06:00",
        close_time="10:00",
        tags=["ramen", "breakfast"],
        occasion_tags={"breakfast"},
    )
    all_day = _minimal_shop(
        name="AllDayRamen",
        open_time="06:00",
        close_time="23:00",
        tags=["ramen", "breakfast"],
        occasion_tags={"breakfast"},
    )
    ranked_sc = [
        RankedShop(shop=all_day, final_score=95.0),
        RankedShop(shop=scarce, final_score=85.0),
    ]
    r_sc = ItinerarySynthesizer.synthesize(
        ranked_sc,
        traffic,
        pref,
        start_time=start5,
        meal_slots=["breakfast"],
        mode=OptimizationMode.BALANCED,
    )
    mids2 = []
    for n in r_sc.nodes:
        mid = getattr(n, "title", "") or ""
        if "·" in mid:
            mids2.append(mid)
    out.append(f"FIRST scarcity {mids2}")

    plain = _minimal_shop(
        name="PlainNight",
        open_time="18:00",
        close_time="23:59",
        tags=["snack"],
        occasion_tags={"late_night"},
    )
    izakaya_n = _minimal_shop(
        name="IzakayaNight",
        open_time="18:00",
        close_time="23:59",
        tags=["izakaya", "late_night"],
        occasion_tags={"late_night", "social"},
    )
    ranked_ln = [
        RankedShop(shop=plain, final_score=95.0),
        RankedShop(shop=izakaya_n, final_score=90.0),
    ]
    r_ln = ItinerarySynthesizer.synthesize(
        ranked_ln,
        traffic,
        pref,
        start_time=datetime(2026, 4, 27, 22, 0, 0),
        meal_slots=["late_night"],
        mode=OptimizationMode.BALANCED,
    )
    mids3 = []
    for n in r_ln.nodes:
        mid = getattr(n, "title", "") or ""
        if "·" in mid:
            mids3.append(mid)
    out.append(f"FIRST late_night {mids3}")

    breakfast_shop6 = _minimal_shop(
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
    ranked_g = [
        RankedShop(shop=breakfast_shop6, final_score=80.0),
        RankedShop(shop=lunch_shop, final_score=90.0),
    ]
    g = GraphBuilder.build_graph(
        ranked=ranked_g,
        traffic=traffic,
        start_time=datetime(2026, 4, 27, 7, 0, 0),
        meal_slots=["breakfast", "lunch"],
        mode=OptimizationMode.BALANCED,
        requested_meal_count=2,
    )
    out.append(f"GRAPH edges {[ (e.from_node_id, e.to_node_id, e.weight) for e in g.edges]}")
    (Path(__file__).resolve().parent / "_inspect_de_out.txt").write_text("\n".join(out), encoding="utf-8")


if __name__ == "__main__":
    main()
