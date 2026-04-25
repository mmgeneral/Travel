"""
Generic Saga Pattern for the Travel Agent
==========================================

Two use cases, one engine:

1. TRANSACTIONAL SAGA — external side effects
   Book Shoraian -> Book Harbs -> Queue Mensuke
   Compensation: cancel the prior booking if a later step fails.

2. CONVERSATIONAL SAGA — internal agent-state edits (dialog rollback)
   User: "swap dinner to Du Xiao Yue"
       -> agent edits preferences, itinerary, venue_selection
   User: "actually never mind, keep Mensuke"
       -> rollback via compensation log

Design goals
------------
- Every step is idempotent on both forward() and compensate()
- Saga log is append-only and JSON-serialisable (local-first, crash-safe)
- Zero dependency on LangGraph or vLLM — unit-testable without GPU
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional


# ──────────────────────────────────────────────────────────────────────────────
# CORE TYPES
# ──────────────────────────────────────────────────────────────────────────────

class StepStatus(str, Enum):
    PENDING     = "pending"
    COMMITTED   = "committed"
    FAILED      = "failed"
    COMPENSATED = "compensated"


@dataclass
class SagaStep:
    """
    One atomic unit of work.
    forward()    : perform the action, return a JSON-serialisable receipt dict
    compensate() : undo the action using the receipt captured at forward() time
    """
    name:           str
    forward:        Callable[[dict], dict]
    compensate:     Callable[[dict], None]
    receipt:        Optional[dict] = None
    status:         StepStatus = StepStatus.PENDING
    error:          Optional[str] = None
    timestamp_fwd:  Optional[float] = None
    timestamp_comp: Optional[float] = None


@dataclass
class SagaLog:
    """
    Append-only execution log — survives restarts when persisted to disk.
    Callable fields (forward/compensate) are NOT serialised; only receipts
    and status are saved, which is enough to replay compensations.
    """
    saga_id:         str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    kind:            str = "transactional"   # or "conversational"
    started:         float = field(default_factory=time.time)
    steps:           list[SagaStep] = field(default_factory=list)
    state_snapshots: list[dict] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps({
            "saga_id": self.saga_id,
            "kind":    self.kind,
            "started": self.started,
            "steps": [{
                "name":    s.name,
                "receipt": s.receipt,
                "status":  s.status.value,
                "error":   s.error,
                "t_fwd":   s.timestamp_fwd,
                "t_comp":  s.timestamp_comp,
            } for s in self.steps],
            "state_snapshots": self.state_snapshots,
        }, ensure_ascii=False, indent=2)


# ──────────────────────────────────────────────────────────────────────────────
# ENGINE
# ──────────────────────────────────────────────────────────────────────────────

class SagaEngine:
    """
    Runs SagaSteps with automatic compensation on failure.
    Also exposes snapshot() / rollback_to() for conversational undo.
    """

    def __init__(self, kind: str = "transactional", persist_path: Optional[str] = None):
        self.log = SagaLog(kind=kind)
        self.persist_path = persist_path

    # ── transactional ────────────────────────────────────────────────────────

    def run(self, steps: list[SagaStep], context: dict) -> tuple[bool, SagaLog]:
        """
        Execute steps in order. On any failure, compensate all previously
        committed steps in reverse order. Returns (success, log).
        """
        self.log.steps = steps

        for i, step in enumerate(steps):
            try:
                step.timestamp_fwd = time.time()
                step.receipt = step.forward(context) or {}
                step.status  = StepStatus.COMMITTED
                self._persist()
            except Exception as e:
                step.status = StepStatus.FAILED
                step.error  = str(e)
                self._persist()
                self._compensate_through(i - 1)
                return False, self.log

        return True, self.log

    def _compensate_through(self, last_committed_idx: int) -> None:
        """Compensate committed steps in reverse order."""
        for step in reversed(self.log.steps[:last_committed_idx + 1]):
            if step.status != StepStatus.COMMITTED:
                continue
            try:
                step.compensate(step.receipt or {})
                step.status = StepStatus.COMPENSATED
                step.timestamp_comp = time.time()
            except Exception as e:
                # Compensation failure — flag for human review, do not re-raise
                step.error = f"{step.error or ''}; comp_failed: {e}"
            finally:
                self._persist()

    # ── conversational ───────────────────────────────────────────────────────

    def snapshot(self, state: dict) -> int:
        """
        Capture a deep copy of agent state before each edit.
        Returns snapshot index — pass it to rollback_to() to undo.
        Call this BEFORE every agent state mutation.
        """
        self.log.state_snapshots.append(json.loads(json.dumps(state)))
        self._persist()
        return len(self.log.state_snapshots) - 1

    def rollback_to(self, snapshot_idx: int) -> dict:
        """
        Restore agent state to snapshot_idx and truncate later snapshots.
        Also compensates any transactional steps that were committed after
        this snapshot (e.g. cancels a booking made during the rolled-back turns).
        """
        if snapshot_idx < 0 or snapshot_idx >= len(self.log.state_snapshots):
            raise IndexError(f"Snapshot {snapshot_idx} out of range "
                             f"(have {len(self.log.state_snapshots)})")

        restored = json.loads(json.dumps(self.log.state_snapshots[snapshot_idx]))
        self.log.state_snapshots = self.log.state_snapshots[:snapshot_idx + 1]

        # Compensate transactional steps committed after the snapshot timestamp
        snap_time = self.log.started + snapshot_idx  # proxy ordering
        for step in reversed(self.log.steps):
            if (step.status == StepStatus.COMMITTED
                    and step.timestamp_fwd
                    and step.timestamp_fwd > snap_time):
                try:
                    step.compensate(step.receipt or {})
                    step.status = StepStatus.COMPENSATED
                    step.timestamp_comp = time.time()
                except Exception as e:
                    step.error = f"rollback comp_failed: {e}"

        self._persist()
        return restored

    # ── persistence (local-first) ────────────────────────────────────────────

    def _persist(self) -> None:
        if not self.persist_path:
            return
        with open(self.persist_path, "w", encoding="utf-8") as f:
            f.write(self.log.to_json())


# ──────────────────────────────────────────────────────────────────────────────
# DEMO 1 — TRANSACTIONAL SAGA
# Simulate booking three venues; Mensuke fails -> auto-cancel Shoraian + Harbs
# ──────────────────────────────────────────────────────────────────────────────

def demo_transactional() -> None:
    def book_shoraian(ctx: dict) -> dict:
        print("  -> Booking Shoraian (phone)...")
        return {"confirmation": "SHR-2026-0424", "party": ctx["party_size"]}

    def cancel_shoraian(receipt: dict) -> None:
        print(f"  <- COMPENSATE: cancel {receipt['confirmation']}")

    def book_harbs(ctx: dict) -> dict:
        print("  -> Reserving Harbs table...")
        return {"confirmation": "HRB-55412"}

    def cancel_harbs(receipt: dict) -> None:
        print(f"  <- COMPENSATE: cancel {receipt['confirmation']}")

    def queue_mensuke(ctx: dict) -> dict:
        raise RuntimeError("Mensuke: network timeout on queue registration")

    def cancel_mensuke(receipt: dict) -> None:
        print(f"  <- COMPENSATE: release queue slot {receipt.get('slot')}")

    steps = [
        SagaStep("book_shoraian", book_shoraian, cancel_shoraian),
        SagaStep("book_harbs",    book_harbs,    cancel_harbs),
        SagaStep("queue_mensuke", queue_mensuke, cancel_mensuke),
    ]

    engine = SagaEngine(kind="transactional", persist_path="/tmp/saga_txn.json")
    ok, log = engine.run(steps, context={"party_size": 2, "date": "2026-04-24"})

    print(f"\n[transactional] saga_id={log.saga_id}  success={ok}")
    for s in log.steps:
        print(f"  {s.name:20s} {s.status.value:12s} {s.error or ''}")


# ──────────────────────────────────────────────────────────────────────────────
# DEMO 2 — CONVERSATIONAL SAGA
# Agent edits state across two turns; user says undo -> rollback to turn 0
# ──────────────────────────────────────────────────────────────────────────────

def demo_conversational() -> None:
    engine = SagaEngine(kind="conversational", persist_path="/tmp/saga_conv.json")

    # Turn 0: initial state
    state = {
        "preferences": {"cuisine": "ramen"},
        "itinerary":   ["Shoraian", "Harbs", "Mensuke"],
    }
    snap0 = engine.snapshot(state)
    print(f"Turn 0  snap#{snap0}: {state['itinerary']}")

    # Turn 1: user says "swap dinner to Du Xiao Yue"
    state["itinerary"][-1] = "Du Xiao Yue"
    state["preferences"]["cuisine"] = "taiwanese"
    snap1 = engine.snapshot(state)
    print(f"Turn 1  snap#{snap1}: {state['itinerary']}")

    # Turn 2: user says "actually keep Mensuke"
    print("\n[user: undo last change]")
    restored = engine.rollback_to(snap0)
    print(f"Restored:  {restored['itinerary']}")
    print(f"Snapshots remaining: {len(engine.log.state_snapshots)}")


# ──────────────────────────────────────────────────────────────────────────────
# INTEGRATION EXAMPLE — wire into agent.py
# ──────────────────────────────────────────────────────────────────────────────
#
# from saga import SagaEngine, SagaStep
#
# _saga = SagaEngine(kind="conversational", persist_path="saga_state.json")
#
# def node_sync(state: AgentState) -> AgentState:
#     snap = _saga.snapshot(dict(state))   # save before every mutation
#     try:
#         state["preferences"] = merge_prefs(state)
#         state["itinerary"]   = rebuild_itinerary(state)
#     except Exception:
#         state = _saga.rollback_to(snap)  # revert on failure
#     return state
#
# # In your LangGraph, add a "rollback" intent route:
# # if intent == "undo": state = _saga.rollback_to(snap_idx)


if __name__ == "__main__":
    print("=" * 60)
    print("DEMO 1 — Transactional saga with auto-compensation")
    print("=" * 60)
    demo_transactional()

    print("\n" + "=" * 60)
    print("DEMO 2 — Conversational saga with dialog rollback")
    print("=" * 60)
    demo_conversational()
