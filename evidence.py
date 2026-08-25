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
        if self.event_type == "bare_rejection" and self.learning:
            raise ValueError("bare_rejection must have learning=False")
        if self.event_type == "clarification_answer":
            if self.question_options is None or len(self.question_options) != 3:
                raise ValueError("clarification_answer requires question_options with 3 items")
            opts = list(self.question_options)
            if "other" not in opts:
                raise ValueError("question_options must contain 'other'")
            taste_idx = None
            context_idx = None
            for name in opts[:2]:
                if name not in FEATURE_NAME_TO_INDEX:
                    raise ValueError(f"Unknown feature {name!r} in question_options")
                idx = FEATURE_NAME_TO_INDEX[name]
                if idx in TASTE_INDICES:
                    taste_idx = idx
                elif idx in CONTEXT_INDICES:
                    context_idx = idx
                else:
                    raise ValueError(f"Feature {name!r} not allowed in clarification pair")
            if taste_idx is None or context_idx is None:
                raise ValueError("clarification question must contain exactly one taste and one context feature")
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
