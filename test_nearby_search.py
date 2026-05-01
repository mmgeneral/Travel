"""Unit tests for NearbySearchTool helpers (no live Google API calls)."""

from shop_planning import NearbySearchTool


def test_haversine_short_distance() -> None:
    kyoto_lat, kyoto_lng = 35.0116, 135.7681
    d_m = NearbySearchTool._haversine_m(kyoto_lat, kyoto_lng, kyoto_lat + 0.01, kyoto_lng)
    assert 900 < d_m < 1200


def test_merge_by_place_id_dedupes() -> None:
    tool = NearbySearchTool(api_key="")
    a = {"place_id": "pid1", "name": "A"}
    b = {"place_id": "pid1", "name": "A dup"}
    c = {"place_id": "pid2", "name": "C"}
    out = tool._merge_by_place_id([a, b], [c])
    assert len(out) == 2


def test_raw_place_matches_must_have_ramen() -> None:
    p = {"name": "Ippudo Ramen", "types": ["restaurant"], "formatted_address": ""}
    assert NearbySearchTool._raw_place_matches_any_must_have(p, {"ramen"})
    assert not NearbySearchTool._raw_place_matches_any_must_have(p, {"izakaya"})


def test_fallback_broad_geo_queries_nonempty() -> None:
    from agent import _fallback_broad_geo_queries

    qs = _fallback_broad_geo_queries("京都 下午茶", "京都", "jp")
    assert len(qs) >= 3
    assert any("京都" in q for q in qs)
