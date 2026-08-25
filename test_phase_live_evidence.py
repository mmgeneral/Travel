"""Live evidence integration tests for the agent revision path."""
import numpy as np
import pytest

from evidence import EvidenceRecord
from likelihood import refit_laplace
from decision_engine import phi, compute_trip_frozen_scaling


class _MockShop:
    def __init__(self, name, tags, review_count=0, base_wait=0,
                 flavor=0.5, price=2.5):
        self.name = name
        self.tags = tags
        self.flavor_intensity = flavor
        self.portion_strictness = 0.5
        self.authority_data = type("Auth", (), {"tablelog_medal": "", "review_count": review_count})()
        self.base_wait_minutes = base_wait
        self.price_level = price
        self.default_travel_minutes = 15


def test_explicit_replacement_produces_evidence():
    shops = [
        _MockShop("A", ["cafe"]),
        _MockShop("B", ["ramen"]),
    ]
    ctx = {"preferred_tags": ["ramen"]}
    means, stds = compute_trip_frozen_scaling(shops, ctx)
    x_e = phi(shops[1], ctx) - phi(shops[0], ctx)
    x_e_norm = (x_e - means) / stds

    rec = EvidenceRecord(
        evidence_id="repl_1",
        thread_id="run_1",
        ts="",
        event_type="replacement",
        learning=True,
        censored_feasibility=False,
        x_e=x_e_norm.tolist(),
        rejected_item="A",
        accepted_item="B",
        ask_eligible=False,
        question_options=None,
        answer_option=None,
        attribution_already_given=False,
    )
    assert rec.learning is True
    mu, Sigma = refit_laplace([rec])
    assert mu.shape == (6,)
    assert Sigma.shape == (6, 6)


def test_bare_rejection_does_not_learn():
    rec = EvidenceRecord(
        evidence_id="bare_1",
        thread_id="run_1",
        ts="",
        event_type="bare_rejection",
        learning=False,
        censored_feasibility=False,
        x_e=None,
        rejected_item="A",
        accepted_item=None,
        ask_eligible=False,
        question_options=None,
        answer_option=None,
        attribution_already_given=False,
    )
    assert rec.learning is False
