from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from shop_planning import AuthorityData, BookingType, FlavorCategory, QueueStrategy, ShopProfile

_CATALOG_DIR = Path(__file__).resolve().parent / "data" / "shop_catalogs"


def _to_profile(raw: dict) -> ShopProfile:
    d = dict(raw)
    d["booking_type"] = BookingType(str(d.get("booking_type", "NONE")))
    d["queue_strategy"] = QueueStrategy(str(d.get("queue_strategy", "PHYSICAL_LINE")))
    d["flavor_category"] = FlavorCategory(str(d.get("flavor_category", "LIGHT")))
    d["authority_data"] = AuthorityData(**dict(d.get("authority_data", {})))
    d["occasion_tags"] = set(d.get("occasion_tags", []) or [])
    d["closed_weekdays"] = {int(x) for x in (d.get("closed_weekdays", []) or [])}
    for key in ("tags", "backup_options", "reservation_channels", "nearby_atm_options", "allowed_dietary_preferences"):
        if key in d:
            d[key] = list(d.get(key) or [])
    for key in ("supported_ethics", "blocked_allergens", "blocked_medical_conditions"):
        if key in d:
            d[key] = set(d.get(key) or [])
    allowed = set(ShopProfile.__dataclass_fields__.keys())
    clean = {k: v for k, v in d.items() if k in allowed}
    return ShopProfile(**clean)


@lru_cache(maxsize=8)
def _load_catalog_tuple(name: str) -> tuple[ShopProfile, ...]:
    p = _CATALOG_DIR / f"{name}.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    return tuple(_to_profile(item) for item in data.get("shops", []))


def load_shop_catalog(name: str) -> list[ShopProfile]:
    return list(_load_catalog_tuple(name))
