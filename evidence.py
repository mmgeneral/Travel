"""
Phase A2 — EvidenceRecord and PreferenceState schemas.

These Pydantic models will be consumed by the likelihood / posterior update
machinery (Phase A3+).  This module intentionally contains **no** math.
"""
from __future__ import annotations

import math
from typing import List, Literal, Optional

from pydantic import BaseModel, field_validator, model_validator

from preference_features import FEATURE_NAMES, FEATURE_NAME_TO_INDEX, TASTE_INDICES, CONTEXT_INDICES


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
    ask_eligible: bool = False

    @field_validator("x_e")
    @classmethod
    def _validate_x_e(cls, v):
        if v is None:
            return None
        if len(v) != 6:
            raise ValueError("x_e must have length 6")
        if not all(isinstance(x, (int, float)) and math.isfinite(float(x)) for x in v):
            raise ValueError("x_e must contain finite numbers")
        return [float(x) for x in v]

    @field_validator("weight")
    @classmethod
    def _validate_weight(cls, v):
        w = float(v)
        if not (0.0 <= w <= 1.0):
            raise ValueError("weight must be in [0,1]")
        return w

    @model_validator(mode="after")
    def _check_event_invariants(self):
        if self.censored_feasibility and self.learning:
            raise ValueError("censored_feasibility=True implies learning must be False")
        if self.event_type == "bare_rejection":
            if self.learning:
                raise ValueError("bare_rejection must have learning=False")
            return self
        if self.learning and self.x_e is None:
            raise ValueError(f"{self.event_type} learning row requires x_e")
        if self.event_type == "clarification_answer":
            if self.question_options is None or len(self.question_options) != 3:
                raise ValueError("clarification_answer requires question_options with 3 items")
            opts = list(self.question_options)
            if opts[2] != "other":
                raise ValueError("question_options must end with 'other'")
            if FEATURE_NAME_TO_INDEX.get(opts[0]) not in TASTE_INDICES:
                raise ValueError("question_options[0] must be a taste-oriented feature")
            if FEATURE_NAME_TO_INDEX.get(opts[1]) not in CONTEXT_INDICES:
                raise ValueError("question_options[1] must be a situational-cost feature")
            if self.answer_option not in {opts[0], opts[1], "other"}:
                raise ValueError("answer_option must be one of the two feature names or 'other'")
        if self.event_type == "explicit_critique" and self.learning:
            if not self.answer_option or self.answer_option not in FEATURE_NAME_TO_INDEX:
                raise ValueError("explicit_critique learning row requires answer_option to be a feature name")
        return self


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

    @field_validator("mu")
    @classmethod
    def _validate_mu(cls, v):
        if len(v) != 6:
            raise ValueError("mu must have length 6")
        if not all(isinstance(x, (int, float)) and math.isfinite(float(x)) for x in v):
            raise ValueError("mu must contain finite numbers")
        return [float(x) for x in v]

    @field_validator("sigma")
    @classmethod
    def _validate_sigma(cls, v):
        if not isinstance(v, list) or len(v) != 6:
            raise ValueError("sigma must be 6x6 list")
        for row in v:
            if not isinstance(row, list) or len(row) != 6:
                raise ValueError("sigma must be 6x6 list")
            if not all(isinstance(x, (int, float)) and math.isfinite(float(x)) for x in row):
                raise ValueError("sigma entries must be finite")
        return v


if __name__ == "__main__":
    # Basic smoke examples using the frozen schema.
    rows = [
        EvidenceRecord(
            evidence_id="r1",
            thread_id="t1",
            ts="2024-01-01T00:00:00",
            event_type="replacement",
            learning=True,
            censored_feasibility=False,
            x_e=[0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
            ask_eligible=False,
        ),
        EvidenceRecord(
            evidence_id="c1",
            thread_id="t1",
            ts="2024-01-01T00:00:00",
            event_type="clarification_answer",
            learning=True,
            censored_feasibility=False,
            x_e=[0.0, 0.0, 1.0, -1.0, 0.0, 0.0],
            question_options=["heaviness", "travel_min", "other"],
            answer_option="heaviness",
            ask_eligible=False,
        ),
        EvidenceRecord(
            evidence_id="k1",
            thread_id="t1",
            ts="2024-01-01T00:00:00",
            event_type="explicit_critique",
            learning=True,
            censored_feasibility=False,
            x_e=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            answer_option="travel_min",
            weight=0.6,
            ask_eligible=False,
        ),
    ]
    for rec in rows:
        print(rec.evidence_id, rec.event_type, "OK")
