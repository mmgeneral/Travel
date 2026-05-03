"""
RetrieverAgent: candidate discovery + LLM-powered retrieval reasoning.

Responsibilities
----------------
* Read ``state["intent"]`` to understand what the user wants.
* Build a candidate pool from the seed catalog (JSON-backed) and the
  dynamic Places API results (NearbySearchTool).
* Ask an LLM (Gemini via LLMRouter, TaskType.RETRIEVAL_REASONING) to
  produce short "why this fits / why not" notes for the candidate pool.
* Detect coverage *gaps* (e.g. city not in seed catalog, meal slots
  uncovered) and surface them so the critic can act on them.
* Return a ``RetrievalReport`` — the agent's deliverable.

Non-responsibilities (stay in later nodes)
------------------------------------------
* Ranking / scoring / filtering → decision_engine / critic
* Producing a final itinerary → synthesizer / node_plan
* Remembering state across calls → everything is read from / written to
  the LangGraph state dict; RetrieverAgent is stateless.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any

from shop_catalog_io import load_shop_catalog
from shop_planning import NearbySearchTool, ShopProfile


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class RetrievalReport:
    city: str
    region: str
    candidates: list[ShopProfile]
    notes: list[str]          # LLM reasoning; one item per notable candidate or theme
    gaps: list[str]           # self-identified coverage gaps for the critic
    seed_count: int = 0
    dynamic_count: int = 0
    query: str = ""

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable snapshot for ``state['retrieval_history']``."""
        return {
            "city": self.city,
            "region": self.region,
            "query": self.query,
            "candidate_names": [c.name for c in self.candidates],
            "notes": self.notes,
            "gaps": self.gaps,
            "seed_count": self.seed_count,
            "dynamic_count": self.dynamic_count,
        }


# ---------------------------------------------------------------------------
# Catalog helpers (no dependency on agent.py to avoid circular imports)
# ---------------------------------------------------------------------------

_SEED_CATALOG_MAP: dict[str, str] = {
    "台北": "taipei",
    "taipei": "taipei",
    "京都": "kyoto",
    "kyoto": "kyoto",
    "東京": "tokyo",
    "tokyo": "tokyo",
    "大阪": "kyoto",   # fallback to kyoto for cities without dedicated catalog
    "osaka": "kyoto",
}


def _load_seed_for_city(city: str, region: str) -> list[ShopProfile]:
    """Return the best available seed catalog for the requested city.

    Falls back to an empty list when the city has no dedicated catalog;
    the gap detector will flag this for the critic.
    """
    key = _SEED_CATALOG_MAP.get(city.lower() if city else "", "")
    if not key:
        # Try the region-level default
        key = "kyoto" if region == "jp" else ("taipei" if region == "tw" else "")
    if not key:
        return []
    try:
        return load_shop_catalog(key)
    except Exception:
        return []


def _has_dedicated_catalog(city: str) -> bool:
    """True when the city has its own JSON catalog file."""
    city_l = (city or "").lower()
    return city_l in {"台北", "taipei", "京都", "kyoto", "東京", "tokyo"}


# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------

_RETRIEVER_SYSTEM = """\
You are a foodie travel research assistant.
Given a user's travel intent and a list of candidate restaurants, provide:
1. Short reasoning notes about the selection (which candidates best match the intent, which less so).
2. Gaps: what is missing from this selection that the user would likely want.

Output ONLY a JSON object — no markdown, no explanation:
{
  "notes": ["<candidate_or_theme>: <1-2 sentence reasoning>", ...],
  "gaps":  ["<gap description>", ...]
}

Rules:
- notes: cover at most 8 notable candidates; be specific about WHY they fit or don't fit the intent.
- gaps: list concrete missing elements (e.g. "No late-night options", "Tokyo catalog is sparse").
- If candidate list is empty, notes=[] and describe gaps thoroughly.
- Do NOT suggest final itineraries or rankings. Only evaluate the pool.
"""


def _build_retriever_prompt(
    intent: dict,
    candidates: list[ShopProfile],
) -> list[dict[str, str]]:
    city = intent.get("city") or "Unknown"
    region = intent.get("region") or "unknown"
    meal_slots = intent.get("meal_slots") or []
    category_tags = intent.get("category_tags") or []
    dietary_hints = intent.get("dietary_hints")
    mode = intent.get("mode") or "balanced"

    intent_desc = (
        f"City: {city} (region={region}), "
        f"mode={mode}, "
        f"meal slots: {', '.join(meal_slots) or 'unspecified'}, "
        f"food categories: {', '.join(category_tags) or 'any'}, "
        f"dietary: {dietary_hints or 'none'}"
    )

    sample = candidates[:20]
    if sample:
        lines = []
        for s in sample:
            tags_str = ", ".join(str(t) for t in list(s.tags)[:5])
            lines.append(
                f"- {s.name} | rating={s.google_rating} | tags=[{tags_str}]"
            )
        cand_block = "\n".join(lines)
    else:
        cand_block = "(no candidates found)"

    user_msg = (
        f"Intent: {intent_desc}\n\n"
        f"Candidate pool ({len(candidates)} shops, showing up to 20):\n{cand_block}"
    )

    return [
        {"role": "system", "content": _RETRIEVER_SYSTEM},
        {"role": "user", "content": user_msg},
    ]


def _parse_llm_notes_gaps(content: str) -> tuple[list[str], list[str]]:
    """Extract notes and gaps from LLM JSON response; tolerates markdown fences."""
    raw = (content or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw).strip()
    try:
        obj = json.loads(raw)
        notes = [str(n) for n in (obj.get("notes") or []) if n]
        gaps = [str(g) for g in (obj.get("gaps") or []) if g]
        return notes, gaps
    except (json.JSONDecodeError, Exception):
        return [], [f"LLM response could not be parsed: {raw[:200]}"]


# ---------------------------------------------------------------------------
# Gap detection (rule-based, before LLM call so LLM can add to them)
# ---------------------------------------------------------------------------

def _detect_structural_gaps(
    city: str,
    region: str,
    meal_slots: list[str],
    category_tags: list[str],
    seed_shops: list[ShopProfile],
    dynamic_shops: list[ShopProfile],
) -> list[str]:
    gaps: list[str] = []

    if not _has_dedicated_catalog(city):
        gaps.append(
            f"No dedicated seed catalog for {city!r}; "
            "results rely on dynamic Places search which may be incomplete."
        )
    elif not seed_shops:
        gaps.append(
            f"Seed catalog for {city!r} is empty or failed to load; "
            "cannot guarantee quality baseline shops."
        )

    all_shops = seed_shops + dynamic_shops
    all_tags: set[str] = set()
    for s in all_shops:
        all_tags.update(str(t).lower() for t in getattr(s, "tags", []))
        all_tags.update(str(t).lower() for t in getattr(s, "occasion_tags", []))

    _SLOT_NEEDS: dict[str, frozenset[str]] = {
        "breakfast": frozenset({"breakfast", "brunch", "morning", "cafe"}),
        "lunch": frozenset({"lunch", "main_meal", "ramen", "quick_meal"}),
        "tea": frozenset({"tea", "dessert", "cafe", "cake", "bakery", "afternoon_tea"}),
        "dinner": frozenset({"dinner", "main_meal", "izakaya", "course"}),
        "late_night": frozenset({"late_night", "izakaya", "ramen", "nightlife"}),
    }
    for slot in meal_slots:
        needed = _SLOT_NEEDS.get(slot, frozenset())
        if needed and not (needed & all_tags):
            gaps.append(
                f"No candidates clearly cover the '{slot}' meal slot — "
                "the pool may need expansion."
            )

    for tag in category_tags:
        if tag.lower() not in all_tags:
            gaps.append(
                f"Requested category '{tag}' not found in any candidate's tags."
            )

    if not all_shops:
        gaps.append("Candidate pool is completely empty; check API keys and catalog files.")

    return gaps


# ---------------------------------------------------------------------------
# RetrieverAgent
# ---------------------------------------------------------------------------

class RetrieverAgent:
    """Stateless retrieval agent.  Create a new instance per call if desired.

    Parameters
    ----------
    llm_router:
        An ``LLMRouter`` instance (or any object with a ``complete`` method
        accepting ``(TaskType, messages)``).  Injected for testability.
    """

    def __init__(self, llm_router: Any) -> None:
        self._router = llm_router

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, state: dict) -> RetrievalReport:
        """Discover candidates and produce reasoning notes for downstream agents."""
        intent = state.get("intent") or {}
        city = intent.get("city") or "京都"
        region = intent.get("region") or "jp"
        meal_slots: list[str] = list(intent.get("meal_slots") or [])
        category_tags: list[str] = list(intent.get("category_tags") or [])
        query = state.get("query") or ""

        # 1. Seed catalog
        seed_shops = _load_seed_for_city(city, region)

        # 2. Dynamic candidates from Places API
        dynamic_shops = self._fetch_dynamic(query, city, region, meal_slots, seed_shops)

        # 3. Rule-based gap detection (before LLM for efficiency)
        structural_gaps = _detect_structural_gaps(
            city, region, meal_slots, category_tags, seed_shops, dynamic_shops
        )

        # 4. Deduplicate seed + dynamic
        candidates = self._dedup(seed_shops, dynamic_shops)

        # 5. LLM reasoning
        notes, llm_gaps = self._llm_reason(intent, candidates)

        # Merge gaps: structural first, then LLM-discovered
        gaps = structural_gaps + [g for g in llm_gaps if g not in structural_gaps]

        return RetrievalReport(
            city=city,
            region=region,
            candidates=candidates,
            notes=notes,
            gaps=gaps,
            seed_count=len(seed_shops),
            dynamic_count=len(dynamic_shops),
            query=query,
        )

    async def arun(self, state: dict) -> RetrievalReport:
        """Async entry: offload sync HTTP/LLM so the LangGraph event loop is not blocked."""
        return await asyncio.to_thread(self.run, state)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _fetch_dynamic(
        self,
        query: str,
        city: str,
        region: str,
        meal_slots: list[str],
        seed_shops: list[ShopProfile],
    ) -> list[ShopProfile]:
        """Call NearbySearchTool and convert results to ShopProfile objects."""
        try:
            tool = NearbySearchTool()
            search_q = f"{query} {city}".strip() if query else city
            places = tool.search_places(city=city, user_query=search_q, limit=40)
        except Exception:
            return []

        shops: list[ShopProfile] = []
        for p in places[:40]:
            try:
                sp = self._place_to_profile(p, region)
                shops.append(sp)
            except Exception:
                continue
        return shops

    @staticmethod
    def _place_to_profile(place: dict, region: str) -> ShopProfile:
        """Minimal ShopProfile from a Places result dict."""
        from shop_planning import BookingType, QueueStrategy

        name = str(place.get("name", "")).strip() or "Unknown"
        rating = float(place.get("rating", 0.0) or 0.0)
        trust = max(0.55, min(0.95, rating / 5.0 if rating > 0 else 0.68))
        types = [str(t).lower() for t in place.get("types", [])]
        tags: list[str] = ["dynamic", "restaurant"]
        if any("cafe" in t for t in types) or "cafe" in name.lower():
            tags.extend(["cafe", "light"])
        if "ramen" in name.lower():
            tags.extend(["ramen", "main_meal"])
        return ShopProfile(
            name=name,
            google_rating=rating,
            trust_score=trust,
            tags=tags,
            booking_type=BookingType.NONE,
            queue_strategy=QueueStrategy.PHYSICAL_LINE,
            close_time="21:00",
            last_call_offset=30,
            is_cash_only=False,
            sns_handle="",
            region=region,
        )

    @staticmethod
    def _dedup(
        seed_shops: list[ShopProfile],
        dynamic_shops: list[ShopProfile],
    ) -> list[ShopProfile]:
        seen: set[str] = set()
        out: list[ShopProfile] = []
        for s in seed_shops + dynamic_shops:
            key = s.name.strip().lower()
            if key and key not in seen:
                seen.add(key)
                out.append(s)
        return out

    def _llm_reason(
        self,
        intent: dict,
        candidates: list[ShopProfile],
    ) -> tuple[list[str], list[str]]:
        """Call LLM for candidate reasoning; returns (notes, gaps)."""
        from llm_router import TaskType

        messages = _build_retriever_prompt(intent, candidates)

        try:
            response = self._router.complete(TaskType.RETRIEVAL_REASONING, messages)
            return _parse_llm_notes_gaps(response.content)
        except Exception as exc:
            return [], [f"LLM reasoning failed: {exc!s}"]
