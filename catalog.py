from shop_catalog_io import load_shop_catalog

from shop_planning import (
    AuthorityData,
    BookingType,
    FlavorCategory,
    QueueStrategy,
    ShopProfile,
)


def _build_shop_catalog() -> list[ShopProfile]:
    """JSON-backed Kyoto seed catalog thin wrapper."""
    return load_shop_catalog("kyoto")


def _build_shop_catalog_taipei() -> list[ShopProfile]:
    """JSON-backed Taipei seed catalog thin wrapper."""
    return load_shop_catalog("taipei")


def _build_shop_catalog_tokyo() -> list[ShopProfile]:
    """JSON-backed Tokyo seed catalog thin wrapper."""
    return load_shop_catalog("tokyo")
