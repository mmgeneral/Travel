"""
Shared TypedDicts for both main.py (simple RAG agent)
and agent.py (autonomous Search→Verify→Audit→Plan agent).
"""
from typing import Annotated, Any, Dict, List, Optional
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

    ``intent`` snapshots follow ``intent_parser.Intent.as_dict()`` keys (JSON-serialisable).
    ``intent_history`` keeps prior snapshots for intent-correction / UX audit trails.
    """
    query:            str
    research_log:     List[str]        # raw authority-domain snippets
    venue_facts:      dict             # per-venue extracted + station-audited facts
    transit_audit:    List[str]        # verified leg-by-leg transit timings
    draft_slots:      List[dict]       # VenueSlot dicts (arrive/depart/notes/warnings)
    itinerary_slots:  List[dict]       # Per-slot plan (shop_name, meal_type, locked state, start_time, duration); written by _node_plan_core, not interchangeable with draft_slots
    critic_report:    List[str]        # Validator + LLM reflexion findings
    final_itinerary:  str              # final markdown output
    version:          int              # itinerary version for optimistic concurrency
    error:            Optional[str]    # non-fatal error accumulator
    intent:           Optional[Dict[str, Any]]
    intent_history:   List[Dict[str, Any]]
    #: End-of-turn checkpoint ids (typically after ``plan``); used by time-travel UX.
    turn_checkpoints: List[str]
