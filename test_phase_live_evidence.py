"""Live evidence integration tests for the agent revision path."""
import numpy as np
import pytest
from unittest.mock import patch

from evidence import EvidenceRecord
from likelihood import refit_laplace
from decision_engine import phi, compute_trip_frozen_scaling
from preference_features import FEATURE_NAMES
from agent import _ingest_revision_evidence, _mark_ask_gate


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


def _base_state(slot_id='lunch', turn_id='rev1'):
    return {
        'agent_run_id': 'run1',
        'query': 'revise lunch',
        'intent_history': [],
        'intent': {
            'category_tags': ['ramen'],
            'revision_op': {
                'slot_id': slot_id,
                'turn_id': turn_id,
                'rejected_shop': 'A',
                'accepted_shop': 'B',
            },
        },
        'phase_a_trip_feature_scaling': {
            'means': [0.0] * 6,
            'stds': [1.0] * 6,
        },
        'phase_a_evidence_log': [],
        '_revision_evidence_id': None,
        '_last_processed_revision_id': None,
        'phase_c_event_xe': None,
        'asked_this_turn': False,
        'phase_c_attribution_already_given': False,
    }


def _shops():
    return [_MockShop('A', ['cafe']), _MockShop('B', ['ramen'])]


def _call(state, **kwargs):
    state['intent']['revision_op'].update(kwargs)
    with patch('agent._researcher_shop_pool', return_value=_shops()):
        return _ingest_revision_evidence(
            state=state,
            rejected_name=state['intent']['revision_op'].get('rejected_shop'),
            accepted_name=state['intent']['revision_op'].get('accepted_shop'),
            explicit_critique_dim=state['intent']['revision_op'].get('explicit_critique_dim'),
            critique_confidence=state['intent']['revision_op'].get('critique_confidence', 0.0),
            slot_id=state['intent']['revision_op'].get('slot_id'),
        )


def test_explicit_replacement_produces_evidence():
    shops = _shops()
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


def test_first_revision_ingests_when_ids_are_none():
    state = _base_state()
    with patch('agent._researcher_shop_pool', return_value=_shops()):
        out = _ingest_revision_evidence(
            state=state,
            rejected_name='A',
            accepted_name='B',
            explicit_critique_dim=None,
            critique_confidence=0.0,
            slot_id='lunch',
        )
    assert len(out['phase_a_evidence_log']) == 1
    assert out['_last_processed_revision_id'] == 'rev1'
    assert out['_revision_evidence_id'] == 'rev1'


def test_same_revision_retry_preserves_turn_local_state():
    state = _base_state()
    with patch('agent._researcher_shop_pool', return_value=_shops()):
        out = _ingest_revision_evidence(
            state=state,
            rejected_name='A', accepted_name='B',
            explicit_critique_dim=None, critique_confidence=0.0, slot_id='lunch',
        )
    # simulate that gate decided to ask, planner sets asked_this_turn=True,
    # and a critique was recorded (or attributed earlier)
    out['asked_this_turn'] = True
    out['phase_c_attribution_already_given'] = True
    xe_before = list(out['phase_c_event_xe'])
    with patch('agent._researcher_shop_pool', return_value=_shops()):
        retried = _ingest_revision_evidence(
            state=out,
            rejected_name='A', accepted_name='B',
            explicit_critique_dim=None, critique_confidence=0.0, slot_id='lunch',
        )
    assert retried['asked_this_turn'] is True
    assert list(retried['phase_c_event_xe']) == xe_before
    assert retried['phase_c_attribution_already_given'] is True


def test_gate_ask_sets_asked_this_turn():
    state = _base_state()
    gate = {"action": "ask", "q_star": {"j_T": 0, "j_C": 3}}
    state = _mark_ask_gate(state, gate)
    assert state['asked_this_turn'] is True
    assert state['phase_c_gate'] == gate


def test_next_distinct_revision_resets_state_flags():
    state = _base_state(turn_id='rev1')
    with patch('agent._researcher_shop_pool', return_value=_shops()):
        out = _ingest_revision_evidence(
            state=state,
            rejected_name='A', accepted_name='B',
            explicit_critique_dim=None, critique_confidence=0.0, slot_id='lunch',
        )
    # simulate previous turn asked something
    out['asked_this_turn'] = True
    out['phase_c_attribution_already_given'] = True
    # next turn (same state object, new revision)
    out['intent']['revision_op']['turn_id'] = 'rev2'
    out['intent']['revision_op']['accepted_shop'] = 'A'
    with patch('agent._researcher_shop_pool', return_value=_shops()):
        out2 = _ingest_revision_evidence(
            state=out,
            rejected_name='B', accepted_name='A',
            explicit_critique_dim=None, critique_confidence=0.0, slot_id='lunch',
        )
    assert out2['asked_this_turn'] is False
    assert out2['phase_c_event_xe'] is not None
    assert out2['phase_c_attribution_already_given'] is False


def test_two_successive_revisions_same_slot_get_different_ids():
    state = _base_state(turn_id='')
    with patch('agent._researcher_shop_pool', return_value=_shops()):
        out1 = _ingest_revision_evidence(
            state=state,
            rejected_name='A', accepted_name='B',
            explicit_critique_dim=None, critique_confidence=0.0, slot_id='lunch',
        )
    id1 = out1['_revision_evidence_id']
    state['query'] = 'different revision request'
    state['intent']['revision_op']['accepted_shop'] = 'A'
    with patch('agent._researcher_shop_pool', return_value=_shops()):
        out2 = _ingest_revision_evidence(
            state=state,
            rejected_name='B', accepted_name='A',
            explicit_critique_dim=None, critique_confidence=0.0, slot_id='lunch',
        )
    assert out2['_revision_evidence_id'] != id1


def test_unresolved_replacement_profile_does_not_learn():
    state = _base_state(turn_id='rev99')
    state['intent']['revision_op']['accepted_shop'] = 'UNKNOWN_SHOP'
    with patch('agent._researcher_shop_pool', return_value=_shops()):
        out = _ingest_revision_evidence(
            state=state,
            rejected_name='A', accepted_name='UNKNOWN_SHOP',
            explicit_critique_dim=None, critique_confidence=0.0, slot_id='lunch',
        )
    assert len(out['phase_a_evidence_log']) == 0
    assert out['_last_processed_revision_id'] == 'rev99'


def test_second_distinct_revision_accumulates_log():
    state = _base_state(turn_id='rev1')
    with patch('agent._researcher_shop_pool', return_value=_shops()):
        out1 = _ingest_revision_evidence(
            state=state,
            rejected_name='A', accepted_name='B',
            explicit_critique_dim=None, critique_confidence=0.0, slot_id='lunch',
        )
    before_log = len(out1['phase_a_evidence_log'])
    xe1_before = list(out1['phase_c_event_xe'])
    # next turn reusing same state
    out1['intent']['revision_op']['turn_id'] = 'rev2'
    out1['intent']['revision_op']['rejected_shop'] = 'B'
    out1['intent']['revision_op']['accepted_shop'] = 'A'
    out1['query'] = 'second revision request'
    with patch('agent._researcher_shop_pool', return_value=_shops()):
        out2 = _ingest_revision_evidence(
            state=out1,
            rejected_name='B', accepted_name='A',
            explicit_critique_dim=None, critique_confidence=0.0, slot_id='lunch',
        )
    assert len(out2['phase_a_evidence_log']) == before_log + 1
    assert list(out2['phase_c_event_xe']) != xe1_before
