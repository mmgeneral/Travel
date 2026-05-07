"""
CriticAgent: dual-perspective (foodie + IC) critique of the retriever's candidate pool.

Responsibilities
----------------
* Read the latest ``RetrievalReport`` from ``state["retrieval_history"][-1]``.
* Reconstruct candidate ``ShopProfile`` objects from the seed catalog and
  ``state["dynamic_shop_pool"]``.
* Score every candidate using ``decision_engine.ScoringEngine`` — both the raw
  weighted score and the fame-damped (underdog_mode=True) score.
* Provide the score table to Claude (via LLMRouter, TaskType.CRITIQUE) so the
  LLM can explain *why* a shop is marketing noise vs. genuine signal.
* Output a ``CritiqueReport`` and append it to ``state["critique_history"]``.

Non-responsibilities
--------------------
* Fetching more data (critic cannot retrieve — only critique what's in state).
* Building the final itinerary (that belongs to node_plan / synthesizer).
* Always agreeing with the retriever — the prompt explicitly instructs Claude to
  find *what the retriever got wrong or missed*.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from decision_engine import (
    RankingEngine,
    ScoringEngine,
    UserPreference,
    WeightProfile,
)
from shop_catalog_io import load_shop_catalog
from shop_planning import AuthorityData, BookingType, QueueStrategy, ShopProfile

from retrieval_service import (
    filter_shop_profiles_by_dietary_exclusions,
    filter_shop_profiles_by_excluded_shop_names,
    normalized_excluded_shop_names_from_intent,
    plan_excluded_frozenset,
)


# ---------------------------------------------------------------------------
# Seed-catalog helper (mirrors agents/retriever.py; no circular import)
# ---------------------------------------------------------------------------

_SEED_CATALOG_MAP: dict[str, str] = {
    "台北": "taipei",
    "taipei": "taipei",
    "京都": "kyoto",
    "kyoto": "kyoto",
    "東京": "tokyo",
    "tokyo": "tokyo",
    "大阪": "kyoto",
    "osaka": "kyoto",
}


def _load_seed_for_city(city: str, region: str) -> list[ShopProfile]:
    key = _SEED_CATALOG_MAP.get((city or "").lower(), "")
    if not key:
        key = "kyoto" if region == "jp" else ("taipei" if region == "tw" else "")
    if not key:
        return []
    try:
        return load_shop_catalog(key)
    except Exception:
        return []


def _pool_dict_to_shop(p: dict, region: str) -> ShopProfile:
    """Reconstruct a minimal ShopProfile from a dynamic_shop_pool entry."""
    name = str(p.get("name", "")).strip() or "Unknown"
    rating = float(p.get("rating", 0.0) or 0.0)
    trust = max(0.55, min(0.95, rating / 5.0 if rating > 0 else 0.68))
    tags: list[str] = ["dynamic", "restaurant"]
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
        region=str(p.get("region", region)),
        latitude=p.get("lat"),
        longitude=p.get("lng"),
        open_time=str(p.get("open_time", "11:00")),
        authority_data=AuthorityData(),
    )


# ---------------------------------------------------------------------------
# Intent → UserPreference
# ---------------------------------------------------------------------------

_DIETARY_ALIAS: dict[str, str] = {
    "vegan": "vegan",
    "vegetarian": "vegetarian",
    "pescatarian": "pescatarian",
    "omnivore": "omnivore",
    "regular": "omnivore",
    "none": "omnivore",
}


def _intent_to_preference(intent: dict) -> UserPreference:
    category_tags = list(intent.get("category_tags") or [])
    dietary_raw = str(intent.get("dietary_hints") or "omnivore").lower().strip()
    dietary = _DIETARY_ALIAS.get(dietary_raw, "omnivore")
    return UserPreference(
        preferred_tags=category_tags,
        avoid_tags=[],
        dietary_preference=dietary,
        max_wait_minutes=35,
        max_total_minutes=120,
        max_budget_impact=0.75,
    )


# ---------------------------------------------------------------------------
# Score helpers
# ---------------------------------------------------------------------------

def _compute_score_tables(
    candidates: list[ShopProfile],
    pref: UserPreference,
) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, bool]]:
    """Return (score_table, fame_damped_table, accolade_table, insider_table)."""
    score_table: dict[str, float] = {}
    fame_damped_table: dict[str, float] = {}
    accolade_table: dict[str, float] = {}
    insider_table: dict[str, bool] = {}

    profile = WeightProfile.trust_first()
    for shop in candidates:
        raw, _ = ScoringEngine.score(shop, pref, profile)
        damped = ScoringEngine.fame_damped_final_score(raw, shop, underdog_mode=True)
        accolade = ScoringEngine.accolade_bonus(shop)
        insider = RankingEngine._qualifies_low_key_bonus(shop)
        score_table[shop.name] = round(raw, 2)
        fame_damped_table[shop.name] = round(damped, 2)
        accolade_table[shop.name] = round(accolade, 2)
        insider_table[shop.name] = insider

    return score_table, fame_damped_table, accolade_table, insider_table


# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------

_CRITIC_SYSTEM = """\
You are a seasoned foodie critic and travel IC (independent consultant).
Your task is to critically review a list of candidate restaurants against the
user's travel intent. You have quantitative scores from our ScoringEngine — use
them as evidence. Do NOT ignore the numbers.

Your dual responsibilities:
1. Foodie lens: Is this list actually interesting? Are high-scored shops
   genuinely good, or are they inflated by awards/hype? A high accolade_bonus
   combined with a much-lower fame_damped score signals marketing noise.
2. IC lens: Are physical constraints satisfied? Are there gaps the retriever
   silently missed (e.g., no breakfast options when user asked for breakfast)?

Critic rules — you MUST follow these:
* Do NOT be a yes-man. Find at least one problem with the pool unless it is
  genuinely perfect.
* "marketing_noise_score >= 0.7 AND accolade_bonus > 10" → flag as over-hyped.
* "is_insider=true" shops are hidden gems — do NOT reject them without cause.
* verdict="deadlock" only when the intent is physically impossible to satisfy
  (e.g., the city has no catalog at all, or requested vegan in all-meat pool).
* requests must reference specific gaps (e.g. "Add local breakfast shops with
  <200 reviews") — never generic ("Find more restaurants").

Output ONLY a JSON object — no markdown, no preamble:
{
  "accepted_names": ["ShopA", "ShopB"],
  "rejected": [{"name": "ShopC", "reason": "<evidence from scores or observations>"}],
  "requests": ["<specific natural-language request for retriever>", ...],
  "verdict": "satisfied" | "request_more" | "deadlock",
  "analysis": "<1-3 sentence overall assessment>"
}
"""


def _build_critique_prompt(
    intent: dict,
    candidates: list[ShopProfile],
    score_table: dict[str, float],
    fame_damped_table: dict[str, float],
    accolade_table: dict[str, float],
    insider_table: dict[str, bool],
    retriever_notes: list[str],
    retriever_gaps: list[str],
    prev_critique: dict,
    iteration: int,
) -> list[dict[str, str]]:
    city = intent.get("city") or "Unknown"
    region = intent.get("region") or "jp"
    meal_slots = intent.get("meal_slots") or []
    category_tags = intent.get("category_tags") or []
    dietary_hints = intent.get("dietary_hints")
    mode = intent.get("mode") or "balanced"

    intent_line = (
        f"city={city} region={region} mode={mode} "
        f"meal_slots=[{', '.join(meal_slots)}] "
        f"categories=[{', '.join(category_tags)}] "
        f"dietary={dietary_hints or 'none'}"
    )

    # Score table (markdown-ish text)
    rows = []
    for shop in candidates[:25]:
        raw = score_table.get(shop.name, 0.0)
        damped = fame_damped_table.get(shop.name, 0.0)
        accolade = accolade_table.get(shop.name, 0.0)
        noise = float(getattr(shop, "marketing_noise_score", None) or 0.45)
        insider = insider_table.get(shop.name, False)
        tags_preview = ", ".join(str(t) for t in list(shop.tags)[:4])
        rows.append(
            f"  {shop.name} | raw={raw} | fame_damped={damped} | "
            f"accolade_bonus={accolade} | noise={noise:.2f} | insider={insider} | "
            f"tags=[{tags_preview}]"
        )
    score_block = "\n".join(rows) if rows else "  (no candidates)"

    prev_block = ""
    if prev_critique:
        prev_verdict = prev_critique.get("verdict", "n/a")
        prev_requests = "; ".join(prev_critique.get("requests_for_retriever") or [])
        prev_block = (
            f"\nPrevious critique (round {iteration - 1}): "
            f"verdict={prev_verdict}, requests={prev_requests or 'none'}"
        )

    gaps_block = (
        "Retriever gaps: " + "; ".join(retriever_gaps)
        if retriever_gaps
        else "Retriever gaps: none reported"
    )
    notes_block = (
        "Retriever notes:\n" + "\n".join(f"  - {n}" for n in retriever_notes[:6])
        if retriever_notes
        else "Retriever notes: none"
    )

    user_msg = (
        f"Intent: {intent_line}\n"
        f"Critique round: {iteration}\n"
        f"{prev_block}\n\n"
        f"Candidate pool ({len(candidates)} shops, showing ≤25):\n"
        f"{score_block}\n\n"
        f"{notes_block}\n"
        f"{gaps_block}"
    )

    return [
        {"role": "system", "content": _CRITIC_SYSTEM},
        {"role": "user", "content": user_msg},
    ]


# ---------------------------------------------------------------------------
# LLM response parsing
# ---------------------------------------------------------------------------

class _CritiqueSchema(BaseModel):
    accepted_names: list[str] = Field(default_factory=list)
    rejected: list[dict[str, str]] = Field(default_factory=list)
    requests: list[str] = Field(default_factory=list)
    verdict: str = "satisfied"
    analysis: str = ""

    @field_validator("verdict")
    @classmethod
    def _check_verdict(cls, v: str) -> str:
        return v if v in {"satisfied", "request_more", "deadlock"} else "request_more"

    @field_validator("rejected", mode="before")
    @classmethod
    def _coerce_rejected(cls, v: Any) -> list[dict[str, str]]:
        if not isinstance(v, list):
            return []
        out = []
        for item in v:
            if isinstance(item, dict):
                out.append({str(k): str(val) for k, val in item.items()})
        return out


def _parse_llm_critique(content: str) -> _CritiqueSchema:
    """Parse LLM JSON response; tolerates markdown fences."""
    raw = (content or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw).strip()
    try:
        obj = json.loads(raw)
        return _CritiqueSchema.model_validate(obj)
    except Exception:
        return _CritiqueSchema(verdict="request_more", analysis=f"parse_error: {raw[:200]}")


# ---------------------------------------------------------------------------
# CritiqueReport
# ---------------------------------------------------------------------------

@dataclass
class CritiqueReport:
    """The critic's deliverable for one critique round."""

    accepted: list[ShopProfile]
    rejected_with_reason: list[tuple[str, str]]   # (shop_name, reason)
    requests_for_retriever: list[str]              # natural-language requests
    verdict: Literal["satisfied", "request_more", "deadlock"]  # type: ignore[assignment]
    score_table: dict[str, float]
    fame_damped_table: dict[str, float]
    accolade_table: dict[str, float]
    llm_analysis: str = ""
    iteration: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "accepted_names": [s.name for s in self.accepted],
            "rejected_with_reason": [list(t) for t in self.rejected_with_reason],
            "requests_for_retriever": self.requests_for_retriever,
            "score_table": self.score_table,
            "fame_damped_table": self.fame_damped_table,
            "accolade_table": self.accolade_table,
            "llm_analysis": self.llm_analysis,
            "iteration": self.iteration,
        }


# ---------------------------------------------------------------------------
# CriticAgent
# ---------------------------------------------------------------------------

class CriticAgent:
    """Stateless critic agent.  Inject llm_router for testability."""

    def __init__(self, llm_router: Any) -> None:
        self._router = llm_router

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, state: dict) -> CritiqueReport:
        intent = state.get("intent") or {}
        city = intent.get("city") or "京都"
        region = intent.get("region") or "jp"
        iteration = int(state.get("research_iteration", 0))

        # 1. Latest retrieval report
        ret_history = list(state.get("retrieval_history") or [])
        ret_report = ret_history[-1] if ret_history else {}
        candidate_names: list[str] = list(ret_report.get("candidate_names") or [])
        retriever_notes: list[str] = list(ret_report.get("notes") or [])
        retriever_gaps: list[str] = list(ret_report.get("gaps") or [])

        # Also fold in researcher's picks so critic sees what researcher selected
        researcher_names: list[str] = list(state.get("researcher_candidate_names") or [])
        all_names = list(dict.fromkeys(candidate_names + researcher_names))  # preserve order, dedup

        # 2. Previous critique (for context / deadlock detection)
        crit_history = list(state.get("critique_history") or [])
        prev_critique = crit_history[-1] if crit_history else {}

        # 3. Reconstruct ShopProfile objects
        xnames = normalized_excluded_shop_names_from_intent(intent)
        seed_shops = filter_shop_profiles_by_dietary_exclusions(
            _load_seed_for_city(city, region), plan_excluded_frozenset(state)
        )
        seed_shops = filter_shop_profiles_by_excluded_shop_names(seed_shops, xnames)
        seed_by_name: dict[str, ShopProfile] = {s.name: s for s in seed_shops}

        dyn_pool = list(state.get("dynamic_shop_pool") or [])
        dyn_by_name: dict[str, dict] = {
            str(p.get("name", "")): p for p in dyn_pool if p.get("name")
        }

        candidates: list[ShopProfile] = []
        seen: set[str] = set()
        for name in all_names:
            if name in seen:
                continue
            seen.add(name)
            if name in seed_by_name:
                candidates.append(seed_by_name[name])
            elif name in dyn_by_name:
                candidates.append(_pool_dict_to_shop(dyn_by_name[name], region))

        # Include the full seed catalog as supplementary context for scoring
        # (candidates list may be sparse on first iteration)
        if not candidates:
            candidates = list(seed_shops[:20])

        # 4. Score with ScoringEngine
        pref = _intent_to_preference(intent)
        score_table, fame_damped_table, accolade_table, insider_table = _compute_score_tables(
            candidates, pref
        )

        # 5. LLM critique (Claude via TaskType.CRITIQUE)
        messages = _build_critique_prompt(
            intent,
            candidates,
            score_table,
            fame_damped_table,
            accolade_table,
            insider_table,
            retriever_notes,
            retriever_gaps,
            prev_critique,
            iteration,
        )
        schema = self._llm_critique(messages, city, len(candidates))

        # 6. Assemble CritiqueReport
        shop_by_name = {s.name: s for s in candidates}
        accepted = [
            shop_by_name[n] for n in schema.accepted_names if n in shop_by_name
        ]
        rejected_with_reason = [
            (str(r.get("name", "")), str(r.get("reason", "")))
            for r in schema.rejected
            if r.get("name")
        ]

        # Ensure verdict consistency: request_more requires non-empty requests
        verdict = schema.verdict
        requests = list(schema.requests)
        if verdict == "request_more" and not requests:
            requests = ["Retriever should expand the candidate pool with more diverse options."]

        return CritiqueReport(
            accepted=accepted,
            rejected_with_reason=rejected_with_reason,
            requests_for_retriever=requests,
            verdict=verdict,  # type: ignore[arg-type]
            score_table=score_table,
            fame_damped_table=fame_damped_table,
            accolade_table=accolade_table,
            llm_analysis=schema.analysis,
            iteration=iteration,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _llm_critique(
        self,
        messages: list[dict[str, str]],
        city: str,
        candidate_count: int,
    ) -> _CritiqueSchema:
        from llm_router import TaskType

        try:
            response = self._router.complete(TaskType.CRITIQUE, messages)
            return _parse_llm_critique(response.content)
        except Exception as exc:
            # Graceful fallback: treat LLM failure as a soft pass
            return _CritiqueSchema(
                verdict="satisfied",
                analysis=f"LLM critique failed ({exc!s}); defaulting to satisfied.",
            )
