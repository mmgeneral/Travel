"""
SynthesizerAgent: on-demand mediator that reads the full R-C transcript and
produces a structured three-section summary.

Design constraints
------------------
* Does NOT write to retrieval_history or critique_history.
* Does NOT call RetrieverAgent or CriticAgent.
* Is NEVER auto-triggered — only runs when the user explicitly requests a pause.
* Uses Gemini (via LLMRouter, TaskType.SYNTHESIS) because it has a large context
  window suitable for summarising multi-round transcripts.

Output: SynthesisReport written to state["synthesis_history"] (append-only).
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field as PydanticField, field_validator

from llm_router import LLMRouter, TaskType
from observability import _get_tracer, record_llm_call


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class SynthesisReport:
    """Three-section transcript summary produced by SynthesizerAgent."""

    consensus: list[str]       # Points R and C both agreed on
    unresolved: list[str]      # Points still in dispute between R and C
    frontier: list[str]        # Gaps R mentioned but that were never addressed
    discussion_freshness: float  # 0–1: how much new ground the last 3 rounds covered
    retrieval_rounds: int = 0  # total retrieval rounds read
    critique_rounds: int = 0   # total critique rounds read
    llm_summary: str = ""      # raw LLM free-text (for Langfuse debugging)

    def as_dict(self) -> dict[str, Any]:
        return {
            "consensus": self.consensus,
            "unresolved": self.unresolved,
            "frontier": self.frontier,
            "discussion_freshness": round(self.discussion_freshness, 3),
            "retrieval_rounds": self.retrieval_rounds,
            "critique_rounds": self.critique_rounds,
            "llm_summary": self.llm_summary,
        }


# ---------------------------------------------------------------------------
# Pydantic schema for LLM output validation
# ---------------------------------------------------------------------------

class _SynthesisSchema(BaseModel):
    consensus: list[str] = PydanticField(default_factory=list)
    unresolved: list[str] = PydanticField(default_factory=list)
    frontier: list[str] = PydanticField(default_factory=list)
    discussion_freshness: float = 0.5

    @field_validator("consensus", "unresolved", "frontier", mode="before")
    @classmethod
    def _ensure_list(cls, v: Any) -> list:
        if isinstance(v, str):
            return [v] if v.strip() else []
        return list(v) if v else []

    @field_validator("discussion_freshness", mode="before")
    @classmethod
    def _clamp_freshness(cls, v: Any) -> float:
        try:
            f = float(v)
        except (TypeError, ValueError):
            return 0.5
        return max(0.0, min(1.0, f))


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_synthesis_prompt(
    retrieval_history: list[dict],
    critique_history: list[dict],
    intent: dict,
) -> list[dict[str, str]]:
    """Construct the Gemini prompt for a three-section synthesis."""
    city = intent.get("city") or "the destination"
    category = ", ".join(intent.get("category_tags") or ["restaurant"])
    query = intent.get("query") or city

    # Summarise retrieval rounds
    r_lines: list[str] = []
    for i, rr in enumerate(retrieval_history, start=1):
        notes = rr.get("notes") or []
        gaps = rr.get("gaps") or []
        candidates = [c.get("name", "?") if isinstance(c, dict) else str(c) for c in (rr.get("candidates") or [])]
        r_lines.append(
            f"Retrieval round {i}: candidates=[{', '.join(candidates[:8])}] "
            f"notes={notes[:3]} gaps={gaps[:3]}"
        )

    # Summarise critique rounds
    c_lines: list[str] = []
    for i, cr in enumerate(critique_history, start=1):
        verdict = cr.get("verdict", "?")
        accepted = [a.get("name", "?") if isinstance(a, dict) else str(a) for a in (cr.get("accepted") or [])]
        rejected_pairs = cr.get("rejected_with_reason") or []
        rejected = [p[0] if isinstance(p, (list, tuple)) and p else str(p) for p in rejected_pairs[:3]]
        requests = cr.get("requests_for_retriever") or []
        analysis = (cr.get("llm_analysis") or "")[:200]
        c_lines.append(
            f"Critique round {i}: verdict={verdict} "
            f"accepted=[{', '.join(accepted[:5])}] "
            f"rejected=[{', '.join(rejected)}] "
            f"requests={requests[:2]} analysis='{analysis}'"
        )

    transcript_text = "\n".join(r_lines + c_lines)

    system = (
        "You are a neutral mediator reviewing a multi-round dialogue between a "
        "Retriever Agent (R) and a Critic Agent (C) about restaurant recommendations. "
        "Your job is to produce a structured JSON summary — not to make new recommendations."
    )

    user = f"""User query: "{query}" (city={city}, type={category})

TRANSCRIPT:
{transcript_text}

Produce a JSON object with exactly these keys:

{{
  "consensus": ["list of points R and C both agreed on — at least 1"],
  "unresolved": ["list of points still contested between R and C — empty list if none"],
  "frontier": ["gaps R mentioned but C never addressed — e.g. missing meal slots, catalog holes"],
  "discussion_freshness": <float 0-1; low (<0.3) if last 3 rounds are near-identical, high (>0.7) if new shops/issues emerged>
}}

Rules:
- consensus: Extract only claims BOTH agents shared. Do not invent.
- unresolved: Only include if C explicitly rejected something R proposed or vice versa.
- frontier: Use R's reported gaps verbatim where possible.
- discussion_freshness: compare round N to round N-1; if candidates and notes barely changed, score < 0.3.
- All lists must contain strings, not objects.
- Output ONLY the JSON object — no markdown fences, no commentary.
"""

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# ---------------------------------------------------------------------------
# Freshness heuristic (fallback if LLM returns implausible value)
# ---------------------------------------------------------------------------

def _compute_freshness_heuristic(retrieval_history: list[dict], critique_history: list[dict]) -> float:
    """Score how much new ground the last 3 rounds produced (0–1)."""
    # Look at the last 3 rounds of each history
    recent_r = retrieval_history[-3:]
    recent_c = critique_history[-3:]

    if not recent_r:
        return 1.0  # Nothing yet — maximum uncertainty = high freshness

    # Collect candidate name sets per round
    candidate_sets: list[frozenset[str]] = []
    for rr in recent_r:
        cands = rr.get("candidates") or []
        names = frozenset(
            (c.get("name", "") if isinstance(c, dict) else str(c)).lower()
            for c in cands
        )
        candidate_sets.append(names)

    if len(candidate_sets) < 2:
        return 0.8  # Only one round — still fresh

    # Jaccard similarity between consecutive rounds
    overlaps = []
    for a, b in zip(candidate_sets, candidate_sets[1:]):
        union = a | b
        if union:
            overlaps.append(len(a & b) / len(union))

    avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0.0

    # Also check if critique verdicts are repeating
    verdict_diversity = len({cr.get("verdict") for cr in recent_c}) / max(1, len(recent_c))

    # High overlap + low verdict diversity → low freshness
    freshness = (1.0 - avg_overlap) * 0.7 + verdict_diversity * 0.3
    return round(max(0.0, min(1.0, freshness)), 3)


# ---------------------------------------------------------------------------
# LLM response parser
# ---------------------------------------------------------------------------

def _parse_synthesis_response(raw: str) -> _SynthesisSchema:
    """Extract JSON from LLM output, tolerating markdown fences."""
    text = raw.strip()
    # Strip markdown code fences if present
    fence = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if fence:
        text = fence.group(1).strip()
    # Find outermost {...}
    brace = re.search(r"\{[\s\S]+\}", text)
    if brace:
        text = brace.group(0)
    try:
        data = json.loads(text)
        return _SynthesisSchema.model_validate(data)
    except Exception:
        return _SynthesisSchema()


# ---------------------------------------------------------------------------
# SynthesizerAgent
# ---------------------------------------------------------------------------

class SynthesizerAgent:
    """On-demand transcript mediator: reads history, never writes to it."""

    def __init__(self, llm_router: LLMRouter) -> None:
        self._router = llm_router

    def run(self, state: dict) -> SynthesisReport:
        retrieval_history: list[dict] = list(state.get("retrieval_history") or [])
        critique_history: list[dict] = list(state.get("critique_history") or [])
        intent: dict = dict(state.get("intent") or {})

        # Fallback freshness (computed before LLM call so we have it on error too)
        heuristic_freshness = _compute_freshness_heuristic(
            retrieval_history, critique_history
        )

        # Short-circuit: if no history at all, return immediately without calling LLM
        if not retrieval_history and not critique_history:
            return SynthesisReport(
                consensus=["No retrieval or critique rounds have run yet."],
                unresolved=[],
                frontier=["Query not started — no gaps to report."],
                discussion_freshness=1.0,
                retrieval_rounds=0,
                critique_rounds=0,
                llm_summary="",
            )

        messages = _build_synthesis_prompt(retrieval_history, critique_history, intent)

        tracer = _get_tracer()
        t0 = time.monotonic()

        def _do_llm_call() -> SynthesisReport:
            resp = self._router.complete(TaskType.SYNTHESIS, messages)
            latency_ms = int((time.monotonic() - t0) * 1000)
            record_llm_call(
                model=resp.model_used,
                tokens_in=resp.tokens_in,
                tokens_out=resp.tokens_out,
                cost=resp.cost_usd,
                latency=latency_ms,
            )
            parsed = _parse_synthesis_response(resp.content)
            llm_freshness = parsed.discussion_freshness
            # Blend LLM freshness (70%) with heuristic (30%) for robustness
            final_freshness = round(llm_freshness * 0.7 + heuristic_freshness * 0.3, 3)
            return SynthesisReport(
                consensus=parsed.consensus or ["No explicit consensus found."],
                unresolved=parsed.unresolved,
                frontier=parsed.frontier,
                discussion_freshness=final_freshness,
                retrieval_rounds=len(retrieval_history),
                critique_rounds=len(critique_history),
                llm_summary=resp.content[:800],
            )

        def _fallback(exc: Exception) -> SynthesisReport:
            return SynthesisReport(
                consensus=["LLM synthesis unavailable — using heuristic fallback."],
                unresolved=self._extract_unresolved_heuristic(critique_history),
                frontier=self._extract_frontier_heuristic(retrieval_history),
                discussion_freshness=heuristic_freshness,
                retrieval_rounds=len(retrieval_history),
                critique_rounds=len(critique_history),
                llm_summary=f"[error: {exc}]",
            )

        if tracer is not None:
            with tracer.start_as_current_span("synthesizer.run") as span:
                span.set_attribute("retrieval_rounds", len(retrieval_history))
                span.set_attribute("critique_rounds", len(critique_history))
                try:
                    result = _do_llm_call()
                    span.set_attribute("llm.model", result.llm_summary[:20])
                    return result
                except Exception as exc:
                    span.set_attribute("llm.error", str(exc))
                    return _fallback(exc)
        else:
            try:
                return _do_llm_call()
            except Exception as exc:
                return _fallback(exc)

    # ------------------------------------------------------------------
    # Heuristic fallbacks (no LLM needed)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_unresolved_heuristic(critique_history: list[dict]) -> list[str]:
        """Surface the latest request_more reasons as unresolved points."""
        for cr in reversed(critique_history):
            requests = cr.get("requests_for_retriever") or []
            if requests:
                return [str(r) for r in requests[:4]]
        return []

    @staticmethod
    def _extract_frontier_heuristic(retrieval_history: list[dict]) -> list[str]:
        """Collect gaps from all retrieval rounds, deduplicated."""
        seen: set[str] = set()
        out: list[str] = []
        for rr in retrieval_history:
            for gap in (rr.get("gaps") or []):
                g = str(gap).strip()
                if g and g not in seen:
                    seen.add(g)
                    out.append(g)
        return out[:6]
