"""Catalog retrieval ingress: deterministic filters before any retrieval / critic / planner hop.

Diet exclusions use :data:`decision_engine.ItinerarySynthesizer.DIETARY_EXCLUDED_TAGS` via the
union frozen set stored on graph state as ``plan_excluded_shop_tags``.
User-specified venue avoidances (`intent.excluded_shops`) are matched by storefront substring / compaction.
Negative category tags (`intent.excluded_tags`, e.g. ``matcha``) drop any shop whose ``tags`` or ``occasion_tags`` intersect — unlike dietary exclusions, which gate only shops tagged under ``DIETARY_EXCLUDED_TAGS``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from decision_engine import ItinerarySynthesizer
from shop_catalog_io import load_shop_catalog
from shop_planning import ShopProfile

# Mirror ``agent._SLOT_TAG_COVERAGE`` so meal-slot gating stays aligned with planner anchors.
_MEAL_SLOT_COVERAGE: dict[str, frozenset[str]] = {
    "breakfast": frozenset({"breakfast", "brunch", "morning", "ramen", "cafe"}),
    "lunch": frozenset({"lunch", "main_meal", "ramen", "quick_meal"}),
    "tea": frozenset({"tea", "dessert", "cafe", "cake", "bakery", "afternoon_tea", "coffee", "refresh"}),
    "dinner": frozenset({"dinner", "main_meal", "ramen", "course", "izakaya", "kaiseki", "sukiyaki"}),
    "late_night": frozenset({"late_night", "izakaya", "ramen", "nightlife", "night_food", "late_open"}),
}


def plan_excluded_frozenset(state: Mapping[str, Any]) -> frozenset[str]:
    """Read ``plan_excluded_shop_tags`` list from LangGraph state as a lowercase tag frozenset."""
    raw = state.get("plan_excluded_shop_tags") or []
    return frozenset(str(x).strip().lower() for x in raw if str(x).strip())


def shop_name_matches_exclusion(shop_name: str, excluded_name: str) -> bool:
    """Return True when ``shop_name`` should be dropped because it matches excluded storefront text."""
    sn = str(shop_name or "").strip()
    ex = str(excluded_name or "").strip()
    if len(ex) < 2 or not sn:
        return False
    if sn == ex:
        return True
    compact_sn = "".join(sn.split()).lower()
    compact_ex = "".join(ex.split()).lower()
    if len(compact_ex) >= 2 and compact_ex in compact_sn:
        return True
    if len(compact_sn) >= 2 and compact_sn in compact_ex:
        return True
    return False


def filter_shop_profiles_by_excluded_shop_names(
    shops: list[ShopProfile], excluded_shop_names: tuple[str, ...] | list[str]
) -> list[ShopProfile]:
    """Remove seed / candidate shops whose ``name`` matches any excluded storefront substring."""
    xs = tuple(dict.fromkeys(str(x).strip() for x in excluded_shop_names if str(x).strip()))
    if not xs:
        return list(shops)
    kept: list[ShopProfile] = []
    for s in shops:
        if any(shop_name_matches_exclusion(s.name, ex) for ex in xs):
            continue
        kept.append(s)
    return kept


def filter_shop_profiles_by_excluded_tags(
    shops: list[ShopProfile],
    excluded_tags: frozenset[str],
) -> list[ShopProfile]:
    """Remove shops whose ``tags`` or ``occasion_tags`` intersect ``excluded_tags``."""
    if not excluded_tags:
        return list(shops)
    banned = frozenset(str(t).strip().lower() for t in excluded_tags if str(t).strip())
    if not banned:
        return list(shops)
    kept: list[ShopProfile] = []
    for s in shops:
        if _tag_bag(s) & banned:
            continue
        kept.append(s)
    return kept


def normalized_excluded_shop_names_from_intent(intent: Mapping[str, Any] | None) -> tuple[str, ...]:
    """Unique non-empty excluded venue strings from ``intent['excluded_shops']`` (order preserved)."""
    if intent is None or not isinstance(intent, Mapping):
        return ()
    raw = intent.get("excluded_shops") or []
    return tuple(dict.fromkeys(str(x).strip() for x in raw if x is not None and str(x).strip()))


def normalized_excluded_tags_from_intent(intent: Mapping[str, Any] | None) -> frozenset[str]:
    """Lowercase normalized tag bans from ``intent['excluded_tags']``."""
    if intent is None or not isinstance(intent, Mapping):
        return frozenset()
    raw = intent.get("excluded_tags") or []
    return frozenset(
        dict.fromkeys(str(x).strip().lower() for x in raw if x is not None and len(str(x).strip()) >= 2)
    )


def filter_shop_profiles_by_dietary_exclusions(
    shops: list[ShopProfile], excluded_shop_tags: frozenset[str]
) -> list[ShopProfile]:
    """Drop shops whose tags ∪ occasion_tags intersect ``DIETARY_EXCLUDED_TAGS`` union (ingress)."""
    if not excluded_shop_tags:
        return list(shops)
    return [s for s in shops if not ItinerarySynthesizer.shop_has_excluded_tag(s, excluded_shop_tags)]


@dataclass(frozen=True)
class DietaryConstraints:
    """Retrieval knobs: exclusions are applied at catalog ingress before softer filters."""

    excluded_shop_tags: frozenset[str] = frozenset()
    appetite_light: bool = False
    excluded_shop_names: tuple[str, ...] = ()
    excluded_tags: frozenset[str] = frozenset()


def catalog_stem_for_city(city: str) -> str:
    raw = (city or "").strip().lower().replace("臺", "台")
    if raw in {"台北", "taipei"}:
        return "taipei"
    if raw in {"東京", "tokyo"}:
        return "tokyo"
    if raw in {"京都", "kyoto"}:
        return "kyoto"
    return "kyoto"


def _tag_bag(shop: ShopProfile) -> set[str]:
    tags = {str(t).lower() for t in (shop.tags or [])}
    occ = {str(t).lower() for t in (shop.occasion_tags or set())}
    return tags | occ


def _semantic_meal_overlap(shop: ShopProfile, slot: str) -> bool:
    base = frozenset(_MEAL_SLOT_COVERAGE.get(slot, frozenset()))
    if slot == "tea":
        base |= frozenset({"snack", "tangyuan", "tang_yuan"})
    elif slot == "late_night":
        base |= frozenset({"snack", "night_snack"})
    if not base:
        return True
    return bool(_tag_bag(shop) & base)


def _matches_any_meal_slot(shop: ShopProfile, norm_slots: list[str]) -> bool:
    return any(_semantic_meal_overlap(shop, s) for s in norm_slots)


def _passes_appetite_light(shop: ShopProfile, appetite_light: bool) -> bool:
    if not appetite_light:
        return True
    ps = float(getattr(shop, "portion_strictness", 0.5))
    return not (ps > 0.9 and not bool(getattr(shop, "has_small_portion", False)))


def _passes_category(shop: ShopProfile, category_tags: list[str]) -> bool:
    need = {t.strip().lower() for t in category_tags if str(t).strip()}
    if not need:
        return True
    bag = _tag_bag(shop)
    return any(t in bag for t in need)


def retrieve_seed_candidates(
    *,
    city: str,
    dietary_constraints: DietaryConstraints,
    category_tags: list[str],
    meal_slots: list[str],
) -> list[ShopProfile]:
    """Load JSON seed catalog for ``city``.

    Applies dietary exclusions (:func:`filter_shop_profiles_by_dietary_exclusions`), then intent tag
    bans (:func:`filter_shop_profiles_by_excluded_tags`), then venue-name bans — then appetite /
    category / meal-slot soft gates with tiered fallback.
    """
    stem = catalog_stem_for_city(city)
    raw = list(load_shop_catalog(stem))
    raw = filter_shop_profiles_by_dietary_exclusions(raw, dietary_constraints.excluded_shop_tags)
    raw = filter_shop_profiles_by_excluded_tags(raw, dietary_constraints.excluded_tags)
    raw = filter_shop_profiles_by_excluded_shop_names(raw, dietary_constraints.excluded_shop_names)
    norm_slots = ItinerarySynthesizer._normalize_slot_sequence(list(meal_slots or []))
    cat_list = [str(x) for x in (category_tags or []) if str(x).strip()]
    appetite = dietary_constraints.appetite_light

    def pipe(
        profiles: list[ShopProfile],
        *,
        use_meals: bool,
        use_category: bool,
    ) -> list[ShopProfile]:
        out: list[ShopProfile] = []
        for p in profiles:
            if not _passes_appetite_light(p, appetite):
                continue
            if use_category and not _passes_category(p, cat_list):
                continue
            if use_meals and norm_slots and not _matches_any_meal_slot(p, norm_slots):
                continue
            out.append(p)
        return out

    filtered = pipe(raw, use_meals=True, use_category=True)
    if not filtered and norm_slots:
        filtered = pipe(raw, use_meals=False, use_category=True)
    if not filtered and cat_list:
        filtered = pipe(raw, use_meals=False, use_category=False)
    if not filtered:
        filtered = pipe(raw, use_meals=False, use_category=False)
    return filtered
