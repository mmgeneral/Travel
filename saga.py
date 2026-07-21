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
import random
import time
import uuid
from pathlib import Path
import tempfile
from threading import Lock
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from acl import ActionOutcome, SagaActionResult
from debug_json import debug_json as _dj


class StepStatus(str, Enum):
    PENDING = "pending"
    COMMITTED = "committed"
    FAILED = "failed"
    COMPENSATED = "compensated"


@dataclass
class SagaStep:
    name: str
    forward: Callable[[dict], dict]
    compensate: Callable[[dict], SagaActionResult | None]
    expected_utility: float = 0.0
    receipt: Optional[dict] = None
    status: StepStatus = StepStatus.PENDING
    error: Optional[str] = None
    outcome: str = ActionOutcome.SUCCESS.value
    semantic_status: str = "NOT_STARTED"
    raw_metadata: dict = field(default_factory=dict)
    trace_id: str = ""
    compensation_attempts: int = 0
    comp_events: list[dict] = field(default_factory=list)
    timestamp_fwd: Optional[float] = None
    timestamp_comp: Optional[float] = None


@dataclass
class DecisionSnapshot:
    timestamp: float
    chosen_option: str
    chosen_reason: str
    backup_candidates: list[dict]
    utility_current: float
    utility_backup_max: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SagaLog:
    saga_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    kind: str = "transactional"
    started: float = field(default_factory=time.time)
    steps: list[SagaStep] = field(default_factory=list)
    state_snapshots: list[dict] = field(default_factory=list)
    decision_snapshots: list[DecisionSnapshot] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(
            {
                "saga_id": self.saga_id,
                "kind": self.kind,
                "started": self.started,
                "steps": [
                    {
                        "name": s.name,
                        "expected_utility": s.expected_utility,
                        "receipt": s.receipt,
                        "status": s.status.value,
                        "error": s.error,
                        "outcome": s.outcome,
                        "semantic_status": s.semantic_status,
                        "trace_id": s.trace_id,
                        "raw_metadata": s.raw_metadata,
                        "t_fwd": s.timestamp_fwd,
                        "t_comp": s.timestamp_comp,
                        "compensation_attempts": s.compensation_attempts,
                        "events": s.comp_events,
                    }
                    for s in self.steps
                ],
                "state_snapshots": self.state_snapshots,
                "decision_snapshots": [
                    {
                        "timestamp": d.timestamp,
                        "chosen_option": d.chosen_option,
                        "chosen_reason": d.chosen_reason,
                        "backup_candidates": d.backup_candidates,
                        "utility_current": d.utility_current,
                        "utility_backup_max": d.utility_backup_max,
                        "metadata": d.metadata,
                    }
                    for d in self.decision_snapshots
                ],
            },
            ensure_ascii=False,
            indent=2,
        )


class RegretMonitor:
    @staticmethod
    def should_rollback(utility_current: float, utility_backup: float) -> bool:
        return utility_current < utility_backup


class SagaEngine:
    _persist_lock = Lock()

    def __init__(
        self,
        kind: str = "transactional",
        persist_path: Optional[str] = None,
        max_retries: int = 3,
        base_backoff_s: float = 1.0,
        jitter_max_s: float = 0.35,
    ):
        self.log = SagaLog(kind=kind)
        self.persist_path = persist_path
        self.max_retries = max_retries
        self.base_backoff_s = base_backoff_s
        self.jitter_max_s = jitter_max_s

    def run(self, steps: list[SagaStep], context: dict) -> tuple[bool, SagaLog]:
        self.log.steps = steps
        for i, step in enumerate(steps):
            regret_probe = context.get("regret_probe")
            if callable(regret_probe):
                decision = regret_probe(step, context) or {}
                utility_current = float(decision.get("utility_current", step.expected_utility))
                utility_backup = float(decision.get("utility_backup", utility_current))
                chosen_option = str(decision.get("chosen_option", step.name))
                chosen_reason = str(decision.get("chosen_reason", "decision_probe"))
                backups = list(decision.get("backup_candidates", []))
                self.snapshot_decision(
                    chosen_option=chosen_option,
                    chosen_reason=chosen_reason,
                    backup_candidates=backups,
                    utility_current=utility_current,
                    utility_backup_max=utility_backup,
                    metadata={"step": step.name},
                )
                if RegretMonitor.should_rollback(utility_current, utility_backup):
                    step.status = StepStatus.FAILED
                    step.outcome = ActionOutcome.FAILED.value
                    step.semantic_status = "REGRET_ROLLBACK_TRIGGERED"
                    step.error = (
                        f"regret-driven rollback: utility_current={utility_current:.2f} "
                        f"< utility_backup={utility_backup:.2f}"
                    )
                    self.rollback_to_last_decision_point(context)
                    self._persist()
                    self._compensate_through(i - 1)
                    return False, self.log
            try:
                step.timestamp_fwd = time.time()
                step.receipt = step.forward(context) or {}
                step.status = StepStatus.COMMITTED
                self._persist()
            except Exception as e:
                step.status = StepStatus.FAILED
                step.error = str(e)
                step.outcome = ActionOutcome.FAILED.value
                step.semantic_status = "FORWARD_FAILED"
                self._persist()
                self._compensate_through(i - 1)
                return False, self.log
        return True, self.log

    def compensate_all(self) -> None:
        self._compensate_through(len(self.log.steps) - 1)

    def _compensate_through(self, last_committed_idx: int) -> None:
        for step in reversed(self.log.steps[: last_committed_idx + 1]):
            if step.status != StepStatus.COMMITTED:
                continue
            self._run_compensation_with_retry(step)
            self._persist()

    def snapshot(self, state: dict) -> int:
        self.log.state_snapshots.append(self._json_clone(state))
        self._persist()
        return len(self.log.state_snapshots) - 1

    def rollback_to(self, snapshot_idx: int) -> dict:
        if snapshot_idx < 0 or snapshot_idx >= len(self.log.state_snapshots):
            raise IndexError(f"Snapshot {snapshot_idx} out of range (have {len(self.log.state_snapshots)})")

        restored = self._json_clone(self.log.state_snapshots[snapshot_idx])
        self.log.state_snapshots = self.log.state_snapshots[: snapshot_idx + 1]

        snap_time = self.log.started + snapshot_idx
        for step in reversed(self.log.steps):
            if step.status == StepStatus.COMMITTED and step.timestamp_fwd and step.timestamp_fwd > snap_time:
                self._run_compensation_with_retry(step)
        self._persist()
        return restored

    def snapshot_decision(
        self,
        chosen_option: str,
        chosen_reason: str,
        backup_candidates: list[dict],
        utility_current: float,
        utility_backup_max: float,
        metadata: Optional[dict[str, Any]] = None,
    ) -> int:
        snap = DecisionSnapshot(
            timestamp=time.time(),
            chosen_option=chosen_option,
            chosen_reason=chosen_reason,
            backup_candidates=backup_candidates,
            utility_current=utility_current,
            utility_backup_max=utility_backup_max,
            metadata=metadata or {},
        )
        self.log.decision_snapshots.append(snap)
        self._persist()
        return len(self.log.decision_snapshots) - 1

    def rollback_to_last_decision_point(self, context: Optional[dict] = None) -> dict:
        if not self.log.decision_snapshots:
            raise IndexError("No decision snapshot available")
        last = self.log.decision_snapshots[-1]
        candidates = list(last.backup_candidates)
        if not candidates:
            restored = {
                "chosen_option": last.chosen_option,
                "chosen_reason": last.chosen_reason,
                "fallback_used": None,
            }
            if context is not None:
                context["decision_restore"] = restored
            return restored

        # Switch to the backup with highest TasteScore/utility.
        best = max(candidates, key=lambda x: float(x.get("taste_score", x.get("utility", 0.0))))
        restored = {
            "chosen_option": last.chosen_option,
            "chosen_reason": last.chosen_reason,
            "fallback_used": best,
        }
        if context is not None:
            context["selected_shop"] = best.get("name", best)
            context["decision_restore"] = restored
        return restored

    def _run_compensation_with_retry(self, step: SagaStep) -> None:
        attempts = 0
        max_attempts = max(1, self.max_retries + 1)

        while attempts < max_attempts:
            attempts += 1
            step.compensation_attempts = attempts
            is_retry = attempts > 1
            try:
                receipt = step.receipt or {}
                receipt["_saga_id"] = self.log.saga_id
                receipt["_step_name"] = step.name
                result = step.compensate(receipt) or SagaActionResult(
                    outcome=ActionOutcome.SUCCESS,
                    semantic_status="COMPENSATED",
                )
            except Exception as e:
                result = SagaActionResult(
                    outcome=ActionOutcome.FAILED,
                    semantic_status="COMPENSATION_EXCEPTION",
                    raw_metadata={"http_status": 0},
                    error_message=str(e),
                )

            step.outcome = result.outcome.value
            step.semantic_status = result.semantic_status
            step.raw_metadata = result.raw_metadata
            step.trace_id = result.trace_id or result.raw_metadata.get("duffel_request_id", "")
            step.comp_events.append(
                {
                    "timestamp": time.time(),
                    "saga_id": self.log.saga_id,
                    "step": step.name,
                    "logical_outcome": result.outcome.value,
                    "physical_http_status": result.raw_metadata.get("http_status", 0),
                    "is_retry": is_retry,
                    "trace_id": step.trace_id,
                    "semantic_status": result.semantic_status,
                }
            )

            if result.outcome == ActionOutcome.SUCCESS:
                step.status = StepStatus.COMPENSATED
                step.timestamp_comp = time.time()
                return
            if result.outcome == ActionOutcome.RETRYABLE and attempts < max_attempts:
                # Exponential backoff schedule: 1s, 2s, 4s (+ jitter)
                backoff_base = self.base_backoff_s * (2 ** (attempts - 1))
                delay = backoff_base + random.uniform(0.0, self.jitter_max_s)
                step.comp_events.append(
                    {
                        "timestamp": time.time(),
                        "saga_id": self.log.saga_id,
                        "step": step.name,
                        "logical_outcome": "RETRY_SCHEDULED",
                        "physical_http_status": result.raw_metadata.get("http_status", 0),
                        "is_retry": True,
                        "trace_id": step.trace_id,
                        "semantic_status": result.semantic_status,
                        "retry_after_s": round(delay, 3),
                    }
                )
                time.sleep(delay)
                continue

            step.error = f"{step.error or ''}; comp_{result.outcome.value.lower()}: {result.error_message or result.semantic_status}"
            return

    def _persist(self) -> None:
        if not self.persist_path:
            return
        payload = self.log.to_json()
        path = Path(self.persist_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with SagaEngine._persist_lock:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(path.parent)) as tf:
                tf.write(payload)
                tf.flush()
                try:
                    import os

                    os.fsync(tf.fileno())
                except Exception:
                    pass
                tmp_name = tf.name
            Path(tmp_name).replace(path)

    @staticmethod
    def _json_clone(payload: dict) -> dict:
        try:
            return json.loads(json.dumps(payload))
        except TypeError as exc:
            raise TypeError(
                f"Saga snapshot requires JSON-serializable primitive state: {exc}. "
                "Normalize state before snapshot (e.g., pydantic model_dump(mode='json'))."
            ) from exc

# ──────────────────────────────────────────────────────────────────────────────
# DEMO 1 — TRANSACTIONAL SAGA
# Simulate booking three venues; Mensuke fails -> auto-cancel Shoraian + Harbs
# ──────────────────────────────────────────────────────────────────────────────

def demo_transactional() -> None:
    def book_shoraian(ctx: dict) -> dict:
        print(_dj("demo_saga_forward", venue="Shoraian", channel="phone"))
        return {"confirmation": "SHR-2026-0424", "party": ctx["party_size"]}

    def cancel_shoraian(receipt: dict) -> None:
        print(_dj("demo_saga_compensate", action="cancel_booking", confirmation=receipt["confirmation"]))

    def book_harbs(ctx: dict) -> dict:
        print(_dj("demo_saga_forward", venue="Harbs", channel="table_reservation"))
        return {"confirmation": "HRB-55412"}

    def cancel_harbs(receipt: dict) -> None:
        print(_dj("demo_saga_compensate", action="cancel_booking", confirmation=receipt["confirmation"]))

    def queue_mensuke(ctx: dict) -> dict:
        raise RuntimeError("Mensuke: network timeout on queue registration")

    def cancel_mensuke(receipt: dict) -> None:
        print(_dj("demo_saga_compensate", action="release_queue_slot", slot=receipt.get("slot")))

    steps = [
        SagaStep("book_shoraian", book_shoraian, cancel_shoraian),
        SagaStep("book_harbs",    book_harbs,    cancel_harbs),
        SagaStep("queue_mensuke", queue_mensuke, cancel_mensuke),
    ]

    engine = SagaEngine(kind="transactional", persist_path="/tmp/saga_txn.json")
    ok, log = engine.run(steps, context={"party_size": 2, "date": "2026-04-24"})

    print(_dj("demo_transactional_done", saga_id=log.saga_id, success=ok))
    for s in log.steps:
        print(_dj("demo_transactional_step", name=s.name, status=s.status.value, error=s.error or None))


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
    print(_dj("demo_conversational_turn", turn=0, snap=snap0, itinerary=state["itinerary"]))

    # Turn 1: user says "swap dinner to Du Xiao Yue"
    state["itinerary"][-1] = "Du Xiao Yue"
    state["preferences"]["cuisine"] = "taiwanese"
    snap1 = engine.snapshot(state)
    print(_dj("demo_conversational_turn", turn=1, snap=snap1, itinerary=state["itinerary"]))

    # Turn 2: user says "actually keep Mensuke"
    print(_dj("demo_conversational_user", message="undo_last_change"))
    restored = engine.rollback_to(snap0)
    print(
        _dj(
            "demo_conversational_restored",
            itinerary=restored["itinerary"],
            snapshots_remaining=len(engine.log.state_snapshots),
        )
    )


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
    print(_dj("cli_demo_banner", demo=1, title="transactional_saga_auto_compensation", width=60))
    demo_transactional()

    print(_dj("cli_demo_banner", demo=2, title="conversational_saga_dialog_rollback", width=60))
    demo_conversational()
