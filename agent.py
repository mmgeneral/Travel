from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

# ── Import the standalone Saga engine ─────────────────────────────────────────
from saga import SagaEngine, SagaStep

# --- TYPES ---
class AgentState(TypedDict):
    query:              str
    research_log:       list[str]
    transit_audit:      list[str]
    final_itinerary:    str
    rollback_occurred:  bool
    saga_snapshot_idx:  int   # conversational rollback pointer

@dataclass(frozen=True)
class TripLeg:
    origin: str
    destination: str
    departure_time: datetime
    arrival_time: datetime
    price: float = 0.0
    seats: int = 10
    race_condition: bool = False

class AtomicCommitFailure(Exception):
    pass


# ──────────────────────────────────────────────────────────────────────────────
# SHARED SAGA ENGINE INSTANCES
# ──────────────────────────────────────────────────────────────────────────────

_txn_saga  = SagaEngine(kind="transactional",  persist_path="/tmp/saga_txn.json")
_conv_saga = SagaEngine(kind="conversational", persist_path="/tmp/saga_conv.json")

# --- UTILS ---
def hhmm(s: str) -> datetime:
    h, m = int(s[:2]), int(s[3:5])
    return datetime(2026, 4, 20, h, m)

# --- MOCK TOOLS ---
class SearchTool:
    @staticmethod
    def get_flights():
        """Mocked flight data showing a precarious last-seat situation on Leg 2."""
        return {
            "TPE-NRT": TripLeg("TPE", "NRT", hhmm("08:00"), hhmm("12:00"), 500.0),
            "NRT-SFO": TripLeg("NRT", "SFO", hhmm("15:00"), hhmm("07:00"), 1200.0, seats=1, race_condition=True)
        }

class TransactionManager:
    def __init__(self):
        self.reserved_legs: list[TripLeg] = []

    def reserve(self, leg: TripLeg):
        """Simulates a soft-lock on a seat with race condition detection."""
        print(f"[RESERVE] Attempting to lock seat on {leg.origin} -> {leg.destination}...")
        
        # Simulate a race condition where another process grabs the last seat between search and reserve
        if leg.race_condition and leg.seats <= 1:
            print(f"  ❌ RACE CONDITION: Process B snatched the last seat on {leg.destination} before we could lock it!")
            raise AtomicCommitFailure(f"Last Seat Race Condition on {leg.destination}")
        
        self.reserved_legs.append(leg)
        print(f"  ✅ Soft-lock acquired for {leg.origin} -> {leg.destination}.")

    def rollback(self):
        """Explicitly releases the hold on previously reserved legs (Compensating Transaction)."""
        print("[ROLLBACK] Transaction Failure. Initiating compensating transactions...")
        for leg in reversed(self.reserved_legs):
            print(f"  🔓 RELEASE: Freeing hold on {leg.origin} -> {leg.destination}")
        self.reserved_legs = []

# --- NODES ---
def node_search(state: AgentState) -> AgentState:
    print("\n[SEARCH] Finding flights TPE → NRT → SFO ...")

    # Conversational snapshot BEFORE any mutation — enables undo later
    snap = _conv_saga.snapshot(dict(state))
    state["saga_snapshot_idx"] = snap

    state["research_log"].append(
        "Found TPE-NRT (available) and UA838 NRT-SFO (1 seat left, high contention)."
    )
    return state

def node_audit(state: AgentState) -> AgentState:
    print("\n[AUDIT] Coordinating atomic reservation via saga.py ...")

    flights = SearchTool.get_flights()
    state["rollback_occurred"] = False

    def reserve_tpe_nrt(ctx: dict) -> dict:
        leg = flights["TPE-NRT"]
        print(f"  [RESERVE] {leg.origin} → {leg.destination} ...")
        return {"leg": f"{leg.origin}-{leg.destination}", "price": leg.price}

    def cancel_tpe_nrt(receipt: dict) -> None:
        print(f"  [RELEASE] Freeing hold on {receipt['leg']}")

    def reserve_nrt_sfo(ctx: dict) -> dict:
        leg = flights["NRT-SFO"]
        print(f"  [RESERVE] {leg.origin} → {leg.destination} ...")
        if leg.race_condition and leg.seats <= 1:
            raise AtomicCommitFailure(
                f"Last-seat race condition on {leg.destination}: "
                "another process grabbed it first."
            )
        return {"leg": f"{leg.origin}-{leg.destination}", "price": leg.price}

    def cancel_nrt_sfo(receipt: dict) -> None:
        print(f"  [RELEASE] Freeing hold on {receipt.get('leg', 'NRT-SFO')}")

    steps = [
        SagaStep("reserve_tpe_nrt", reserve_tpe_nrt, cancel_tpe_nrt),
        SagaStep("reserve_nrt_sfo", reserve_nrt_sfo, cancel_nrt_sfo),
    ]

    ok, log = _txn_saga.run(steps, context={"date": "2026-04-24"})

    if not ok:
        failed = next((s for s in log.steps if s.error), None)
        reason = failed.error if failed else "unknown"
        print(f"\n[CRITICAL] Saga failed: {reason}")
        state["rollback_occurred"] = True
        state["transit_audit"].append(reason)
    else:
        print("[SUCCESS] All legs reserved atomically.")

    return state

def node_plan(state: AgentState) -> AgentState:
    print("\n[PLAN] Generating outcome report ...")

    report  = "## 🏁 SAGA PATTERN DEMO — Last-Seat Race Condition (TPE → NRT → SFO)\n\n"
    report += f"**Conversational snapshot index:** `{state['saga_snapshot_idx']}` "
    report += "(call `rollback_to(idx)` to undo this entire planning turn)\n\n"

    if state["rollback_occurred"]:
        reason = state["transit_audit"][0] if state["transit_audit"] else "unknown"
        report += "### ❌ TRANSACTION STATUS: ABORTED\n"
        report += f"**Reason:** {reason}\n\n"
        report += "| | Naive outcome | Saga outcome |\n"
        report += "|---|---|---|\n"
        report += "| Final state | Orphaned TPE-NRT booking | **Clean slate** |\n"
        report += "| Financial risk | $500 non-refundable | **$0 lost** |\n"
        report += "| Consistency | Broken | Strong eventual |\n"
        report += "| Recovery | Manual intervention | **Automatic rollback** |\n\n"
        report += "#### Execution trace\n"
        report += "1. `RESERVE(TPE-NRT)` → ✅ hold acquired\n"
        report += "2. `RESERVE(NRT-SFO)` → ❌ race condition\n"
        report += "3. `COMPENSATE(TPE-NRT)` → ✅ seat returned to inventory\n"
    else:
        report += "### ✅ TRANSACTION STATUS: SUCCESS\n"
        report += "All legs reserved atomically — itinerary locked.\n"

    report += "\n---\n"
    report += "### 💬 Conversational rollback\n"
    report += "Snapshot taken in `node_search` before any state mutation.\n"
    report += "To undo: `restored = _conv_saga.rollback_to(state['saga_snapshot_idx'])`\n"
    report += "Both agent state and committed bookings are reversed.\n"

    state["final_itinerary"] = report
    return state

# --- GRAPH ---
def build_graph():
    g = StateGraph(AgentState)
    g.add_node("search", node_search)
    g.add_node("audit", node_audit)
    g.add_node("plan", node_plan)
    
    g.set_entry_point("search")
    g.add_edge("search", "audit")
    g.add_edge("audit", "plan")
    g.add_edge("plan", END)
    
    return g.compile()

if __name__ == "__main__":
    initial_state: AgentState = {
        "query":             "Book TPE to SFO via NRT (UA838)",
        "research_log":      [],
        "transit_audit":     [],
        "final_itinerary":   "",
        "rollback_occurred": False,
        "saga_snapshot_idx": -1,
    }

    result = build_graph().invoke(initial_state)

    print("\n" + "=" * 70)
    print("  FINAL AGENT RESPONSE")
    print("=" * 70)
    print(result["final_itinerary"])

    # Demo: conversational rollback
    print("=" * 70)
    print("  DEMO — undo the planning turn (conversational rollback)")
    print("=" * 70)
    snap_idx = result["saga_snapshot_idx"]
    if snap_idx >= 0:
        restored = _conv_saga.rollback_to(snap_idx)
        print(f"  Restored keys:          {list(restored.keys())}")
        print(f"  research_log cleared:   {restored['research_log'] == []}")
        print(f"  Snapshots remaining:    {len(_conv_saga.log.state_snapshots)}")
    print("=" * 70)
