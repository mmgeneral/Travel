from __future__ import annotations

from agent import _build_shop_catalog, _build_shop_catalog_taipei, _build_shop_catalog_tokyo


def test_catalog_files_load() -> None:
    assert len(_build_shop_catalog()) >= 1
    assert len(_build_shop_catalog_tokyo()) >= 1
    assert len(_build_shop_catalog_taipei()) >= 12


def test_taipei_required_shops_present() -> None:
    names = {s.name for s in _build_shop_catalog_taipei()}
    assert "青島東路豆漿大王" in names
    assert "永康街高家莊" in names
    assert "欣葉台菜創始店" in names
    assert "世盛 1955" in names


def test_taipei_seed_has_complete_geo_fields() -> None:
    shops = _build_shop_catalog_taipei()
    for s in shops:
        assert s.latitude is not None
        assert s.longitude is not None
        assert isinstance(s.closed_weekdays, set)
        assert bool(s.flavor_vector)


def test_taipei_has_one_third_civilian_gems() -> None:
    shops = _build_shop_catalog_taipei()
    gems = [s for s in shops if 4.0 <= float(s.google_rating or 0.0) <= 4.3]
    assert len(gems) >= len(shops) // 3
"""Catalog-level schedule fixture tests."""

from agent import _build_shop_catalog


def test_shoraian_has_explicit_open_time():
    shops = _build_shop_catalog()
    shoraian = next(s for s in shops if s.name == "松籟庵")
    assert shoraian.open_time == "11:00"


def test_asa_ramen_breakfast_fixture_exists():
    shops = _build_shop_catalog()
    asa_ramen = next(s for s in shops if s.name == "朝拉麵")
    assert asa_ramen.open_time == "06:00"
    assert "breakfast" in asa_ramen.tags
