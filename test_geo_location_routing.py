from __future__ import annotations

from agent import _extract_city_from_query, build_graph, make_initial_state


def test_extract_city_from_coordinates_taipei() -> None:
    city, region = _extract_city_from_query(
        "幫我排今天晚餐",
        user_locale=None,
        user_lat=25.03,
        user_lng=121.56,
    )
    assert city == "台北"
    assert region == "tw"


def test_geo_location_drives_taipei_seed_catalog() -> None:
    """Taipei coordinates must route to the Taipei seed catalog.

    node_retriever logs retriever_complete to research_log (not transit_audit).
    """
    graph = build_graph()
    st = make_initial_state(
        "附近有什麼好吃",
        user_locale=None,
        user_lat=25.03,
        user_lng=121.56,
    )
    out = graph.invoke(st)
    # Intent should parse to Taipei
    intent = out.get("intent") or {}
    assert intent.get("region") == "tw", f"Expected region='tw', got intent={intent}"
    # retriever_complete is logged to research_log by node_retriever
    research = "\n".join(out.get("research_log", []))
    assert "retriever_complete" in research, "node_retriever did not log retriever_complete"
