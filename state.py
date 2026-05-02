"""
Shared TypedDicts for both main.py (simple RAG agent)
and agent.py (autonomous Search→Verify→Audit→Plan agent).
"""
from typing import Annotated, Any, List, Optional
from typing_extensions import TypedDict


class PreferenceState(TypedDict):
    """Simple RAG agent state (main.py)."""
    preferences: dict
    itinerary:   str
    messages:    Annotated[List[dict], "The conversation history"]
    sync_status: str


class AgentState(TypedDict):
    """
    Full autonomous agent state (agent.py).
    Each field is written by exactly one node and read by all downstream nodes.
    """
    query:            str
    research_log:     List[str]        # raw authority-domain snippets
    venue_facts:      dict             # per-venue extracted + station-audited facts
    transit_audit:    List[str]        # verified leg-by-leg transit timings
    draft_slots:      List[dict]       # VenueSlot dicts (arrive/depart/notes/warnings)
    critic_report:    List[str]        # Validator + LLM reflexion findings
    final_itinerary:  str              # final markdown output
    version:          int              # itinerary version for optimistic concurrency
    error:            Optional[str]    # non-fatal error accumulator
