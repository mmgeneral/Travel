"""
Phase A2 — EvidenceRecord and PreferenceState schemas.

These Pydantic models will be consumed by the likelihood / posterior update
machinery (Phase A3+).  This module intentionally contains **no** math.
"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel


class EvidenceRecord(BaseModel):
    """
    One event that can change the posterior.

    event_type:
        - replacement          : user replaces a proposed shop
        - explicit_critique    : user says “this is bad because …”
        - clarification_answer : user answers a prompted question
        - bare_rejection       : user says “no” without additional info
    """
    evidence_id: str
    thread_id: str
    turn_checkpoint_id: Optional[str] = None
    ts: str
    event_type: Literal[
        "replacement",
        "explicit_critique",
        "clarification_answer",
        "bare_rejection",
    ]
    learning: bool
    censored_feasibility: bool
    rejected_item: Optional[str] = None
    accepted_item: Optional[str] = None
    slot_id: Optional[str] = None
    x_e: Optional[List[float]] = None          # length 6
    question_options: Optional[List[str]] = None
    answer_option: Optional[str] = None
    weight: float = 1.0                        # spontaneous critique's ρ
    anchor_evidence_id: Optional[str] = None
    utterance_excerpt: str = ""
    ask_eligible: bool


class PreferenceState(BaseModel):
    """
    The current posterior state for a single thread.
    """
    thread_id: str
    mu: List[float]                          # length 6
    sigma: List[List[float]]                 # 6×6 dense covariance
    n_events: int
    posterior_version: int
    asked_this_turn: bool


if __name__ == "__main__":
    # ---- validation for A2 ----
    examples = [
        dict(event_type="replacement", learning=True, question_options=None, answer_option=None),
        dict(event_type="explicit_critique", learning=True, question_options=None, answer_option=None),
        dict(event_type="clarification_answer", learning=True, question_options=["較清淡", "較重口味"], answer_option="較清淡"),
        dict(event_type="bare_rejection", learning=False, question_options=None, answer_option=None),
    ]

    for idx, ex in enumerate(examples, start=1):
        rec = EvidenceRecord(
            evidence_id=f"test-{idx}",
            thread_id="thread-1",
            ts="2024-01-01T00:00:00",
            event_type=ex["event_type"],
            learning=ex["learning"],
            censored_feasibility=False,
            question_options=ex["question_options"],
            answer_option=ex["answer_option"],
            ask_eligible=True,
        )
        if rec.event_type == "bare_rejection":
            assert rec.learning is False, "bare_rejection must set learning=False"
        else:
            assert rec.learning is True, f"{rec.event_type} must set learning=True"
        if rec.question_options is None:
            assert rec.event_type != "clarification_answer", (
                "clarification_answer must be prompted (question_options non-None)"
            )
        else:
            assert rec.event_type == "clarification_answer", (
                "only clarification_answer may carry question_options"
            )
        print(
            f"OK  event={rec.event_type:<22} "
            f"learning={rec.learning} "
            f"prompted={rec.question_options is not None}"
        )
