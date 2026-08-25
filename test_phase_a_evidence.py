"""Phase A2 schema tests."""
import pytest
from pydantic import ValidationError

from evidence import EvidenceRecord, PreferenceState


def _base(overrides=None):
    d = {
        "evidence_id": "e1",
        "thread_id": "t1",
        "ts": "2024-01-01T00:00:00",
        "event_type": "replacement",
        "learning": True,
        "censored_feasibility": False,
        "x_e": [1.0]*6,
        "ask_eligible": False,
    }
    if overrides:
        d.update(overrides)
    return d


def test_xe_length_rejected():
    with pytest.raises(ValidationError):
        EvidenceRecord(**_base({"x_e": [1.0]*5}))


def test_xe_nonfinite_rejected():
    with pytest.raises(ValidationError):
        EvidenceRecord(**_base({"x_e": [1.0, float("nan"), 0.0, 0, 0, 0]}))


def test_mu_length_rejected():
    with pytest.raises(ValidationError):
        PreferenceState(thread_id="t1", mu=[1.0]*5, sigma=[[0.0]*6 for _ in range(6)], n_events=0, posterior_version=0, asked_this_turn=False)


def test_sigma_shape_rejected():
    with pytest.raises(ValidationError):
        PreferenceState(thread_id="t1", mu=[1.0]*6, sigma=[[0.0]*5 for _ in range(6)], n_events=0, posterior_version=0, asked_this_turn=False)


def test_weight_out_of_range_rejected():
    with pytest.raises(ValidationError):
        EvidenceRecord(**_base({"event_type": "explicit_critique", "answer_option": "travel_min", "weight": 1.2}))


def test_bare_rejection_learning_true_rejected():
    with pytest.raises(ValidationError):
        EvidenceRecord(**_base({"event_type": "bare_rejection", "learning": True, "x_e": None}))


def test_clarification_two_taste_features_rejected():
    with pytest.raises(ValidationError):
        EvidenceRecord(**_base({
            "event_type": "clarification_answer",
            "question_options": ["cuisine_match", "heaviness", "other"],
            "answer_option": "heaviness",
        }))


def test_critique_valid_feature_name_ok():
    rec = EvidenceRecord(**_base({
        "event_type": "explicit_critique",
        "answer_option": "travel_min",
        "weight": 0.7,
    }))
    assert rec.answer_option == "travel_min"
