"""Agent-side DP overlays (distinct shops, slot-tag affinity) — decision_engine untouched."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from decision_engine import GraphEdge, GraphNode, SpatioTemporalGraph
from shop_planning import MockTrafficProvider

from agent import agent_dp_find_optimal_path_no_shop_repeat


@pytest.fixture()
def brunch_traffic():
    return MockTrafficProvider()


def _node(
    node_id: str,
    shop: str,
    slot_index: int,
    start_hour: float,
    final_score: float,
    *,
    tags: tuple[str, ...] = (),
    dur_m: int = 40,
) -> GraphNode:
    base_day = datetime(2026, 5, 2, 0, 0, 0)
    st = base_day.replace(hour=7, minute=0) + timedelta(minutes=int(start_hour * 60))
    en = st + timedelta(minutes=dur_m)
    ideal = dur_m + 15
    return GraphNode(
        node_id=node_id,
        shop_name=shop,
        slot_index=slot_index,
        start_time=st,
        end_time=en,
        final_score=final_score,
        tags=tags,
        ideal_duration_minutes=ideal,
        actual_duration_minutes=dur_m,
    )


def test_dp_path_rejects_revisiting_same_shop_name(brunch_traffic: MockTrafficProvider):
    """High-score duplicate slot must lose to feasible no-repeat continuation."""
    _ = brunch_traffic
    n_dup0 = _node("d0", "DupRamen", 0, 0, 500.0, tags=("ramen",))
    n_mid1 = _node("a1", "MidA", 1, 5, 300.0, tags=("lunch",))
    n_mid2 = _node("b2", "MidB", 2, 10, 300.0, tags=("tea",))
    n_dup3 = _node("d3", "DupRamen", 3, 15, 600.0, tags=("ramen", "dinner"))
    n_alt3 = _node("c3", "AltPlain", 3, 15, 200.0, tags=("dinner",))
    nodes = [n_dup0, n_mid1, n_mid2, n_dup3, n_alt3]
    edges = [
        GraphEdge("d0", "a1", 1.0),
        GraphEdge("a1", "b2", 1.0),
        GraphEdge("b2", "d3", 1.0),
        GraphEdge("b2", "c3", 1.0),
    ]
    graph = SpatioTemporalGraph(nodes=nodes, edges=edges, debug_traces=[])
    setattr(
        graph,
        "_agent_meal_slot_names_for_dp",
        ["breakfast", "lunch", "tea", "dinner"],
    )
    path = agent_dp_find_optimal_path_no_shop_repeat(graph, required_length=4, solver_audit_log=None)
    names = [n.shop_name for n in path]
    assert len(names) == len(set(names))
    assert names == ["DupRamen", "MidA", "MidB", "AltPlain"]


def test_dp_breakfast_prefers_breakfast_tagged_shop(brunch_traffic: MockTrafficProvider):
    _ = brunch_traffic
    n_ramen = _node(
        "r0",
        "RamenEarly",
        0,
        0,
        120.0,
        tags=("ramen", "lunch"),
        dur_m=30,
    )
    n_cafe = _node(
        "c0",
        "CafeMorn",
        0,
        0,
        100.0,
        tags=("breakfast", "cafe"),
        dur_m=30,
    )
    n_lunch = _node(
        "l1",
        "LunchLate",
        1,
        4,
        80.0,
        tags=("lunch",),
        dur_m=45,
    )
    nodes = [n_ramen, n_cafe, n_lunch]
    edges = [GraphEdge("r0", "l1", 1.0), GraphEdge("c0", "l1", 1.0)]
    graph = SpatioTemporalGraph(nodes=nodes, edges=edges, debug_traces=[])
    setattr(graph, "_agent_meal_slot_names_for_dp", ["breakfast", "lunch"])
    path = agent_dp_find_optimal_path_no_shop_repeat(graph, required_length=2, solver_audit_log=None)
    assert path[0].shop_name == "CafeMorn"


def test_dp_late_night_prefers_late_night_tag(brunch_traffic: MockTrafficProvider):
    _ = brunch_traffic
    n_early = _node(
        "e4",
        "EarlyBird",
        4,
        20,
        90.0,
        tags=("dinner",),
        dur_m=25,
    )
    n_ln = _node(
        "n4",
        "NightSnack",
        4,
        20,
        80.0,
        tags=("late_night", "ramen"),
        dur_m=25,
    )
    n_prev = _node(
        "d3",
        "Dinn",
        3,
        15,
        70.0,
        tags=("dinner",),
        dur_m=40,
    )
    nodes = [n_prev, n_early, n_ln]
    edges = [GraphEdge("d3", "e4", 1.0), GraphEdge("d3", "n4", 1.0)]
    graph = SpatioTemporalGraph(nodes=nodes, edges=edges, debug_traces=[])
    setattr(
        graph,
        "_agent_meal_slot_names_for_dp",
        ["breakfast", "lunch", "tea", "dinner", "late_night"],
    )
    path2 = agent_dp_find_optimal_path_no_shop_repeat(graph, required_length=2, solver_audit_log=None)
    assert len(path2) == 2
    assert path2[-1].shop_name == "NightSnack"
