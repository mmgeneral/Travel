"""Live evidence integration tests for the agent revision path."""
import numpy as np
import pytest
from unittest.mock import patch

from evidence import EvidenceRecord
from likelihood import refit_laplace
from decision_engine import phi, compute_trip_frozen_scaling
from preference_features import FEATURE_NAMES
from agent import _ingest_revision_evidence


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


def test_same_revision_id_twice_appends_only_once():
    state = _base_state()
    with patch('agent._researcher_shop_pool', return_value=_shops()):
        first = _ingest_revision_evidence(
            state=state,
            rejected_name='A', accepted_name='B',
            explicit_critique_dim=None, critique_confidence=0.0, slot_id='lunch',
        )
        before = len(first['phase_a_evidence_log'])
        second = _ingest_revision_evidence(
            state=first,
            rejected_name='A', accepted_name='B',
            explicit_critique_dim=None, critique_confidence=0.0, slot_id='lunch',
        )
    assert len(second['phase_a_evidence_log']) == before


def test_second_distinct_revision_overwrites_phase_c_event_xe():
    state = _base_state(turn_id='rev1')
    with patch('agent._researcher_shop_pool', return_value=_shops()):
        out1 = _ingest_revision_evidence(
            state=state,
            rejected_name='A', accepted_name='B',
            explicit_critique_dim=None, critique_confidence=0.0, slot_id='lunch',
        )
        xe1 = list(out1['phase_c_event_xe'])
        state2 = _base_state(turn_id='rev2')
        state2['intent']['revision_op']['accepted_shop'] = 'A'
        with patch('agent._researcher_shop_pool', return_value=_shops()):
            out2 = _ingest_revision_evidence(
                state=state2,
                rejected_name='B', accepted_name='A',
                explicit_critique_dim=None, critique_confidence=0.0, slot_id='lunch',
            )
        xe2 = list(out2['phase_c_event_xe'])
    assert xe1 != xe2


def test_replacement_plus_critique_appends_both():
    state = _base_state()
    state['intent']['revision_op']['explicit_critique_dim'] = 0
    state['intent']['revision_op']['critique_confidence'] = 0.7
    with patch('agent._researcher_shop_pool', return_value=_shops()):
        out = _ingest_revision_evidence(
            state=state,
            rejected_name='A', accepted_name='B',
            explicit_critique_dim=0, critique_confidence=0.7, slot_id='lunch',
        )
    types = [r['event_type'] for r in out['phase_a_evidence_log']]
    assert types.count('replacement') == 1
    assert types.count('explicit_critique') == 1


def test_bare_rejection_plus_critique_appends_both():
    state = _base_state()
    state['intent']['revision_op']['accepted_shop'] = None
    state['intent']['revision_op']['explicit_critique_dim'] = 4
    state['intent']['revision_op']['critique_confidence'] = 0.5
    with patch('agent._researcher_shop_pool', return_value=_shops()):
        out = _ingest_revision_evidence(
            state=state,
            rejected_name='A', accepted_name=None,
            explicit_critique_dim=4, critique_confidence=0.5, slot_id='lunch',
        )
    types = [r['event_type'] for r in out['phase_a_evidence_log']]
    assert types.count('bare_rejection') == 1
    assert types.count('explicit_critique') == 1


def test_critique_answer_option_matches_feature_names():
    state = _base_state()
    state['intent']['revision_op']['accepted_shop'] = None
    state['intent']['revision_op']['explicit_critique_dim'] = 3
    with patch('agent._researcher_shop_pool', return_value=_shops()):
        out = _ingest_revision_evidence(
            state=state,
            rejected_name='A', accepted_name=None,
            explicit_critique_dim=3, critique_confidence=0.8, slot_id='lunch',
        )
    crit = [r for r in out['phase_a_evidence_log'] if r['event_type'] == 'explicit_critique'][0]
    assert crit['answer_option'] == FEATURE_NAMES[3]


def test_critique_weight_equals_rho():
    state = _base_state()
    state['intent']['revision_op']['accepted_shop'] = None
    state['intent']['revision_op']['explicit_critique_dim'] = 2
    rho = 0.45
    with patch('agent._researcher_shop_pool', return_value=_shops()):
        out = _ingest_revision_evidence(
            state=state,
            rejected_name='A', accepted_name=None,
            explicit_critique_dim=2, critique_confidence=rho, slot_id='lunch',
        )
    crit = [r for r in out['phase_a_evidence_log'] if r['event_type'] == 'explicit_critique'][0]
    assert crit['weight'] == pytest.approx(rho)


def test_posterior_changes_only_for_learning_rows():
    state = _base_state()
    with patch('agent._researcher_shop_pool', return_value=_shops()):
        out = _ingest_revision_evidence(
            state=state,
            rejected_name='A', accepted_name='B',
            explicit_critique_dim=None, critique_confidence=0.0, slot_id='lunch',
        )
    mu_after = np.asarray(out['phase_a_posterior_mu'])
    assert not np.allclose(mu_after, np.zeros(6))
    # Bare rejection alone should not change posterior
    state2 = _base_state(turn_id='rev2')
    state2['intent']['revision_op']['accepted_shop'] = None
    before_mu = out['phase_a_posterior_mu']
    with patch('agent._researcher_shop_pool', return_value=_shops()):
        out2 = _ingest_revision_evidence(
            state=state2,
            rejected_name='A', accepted_name=None,
            explicit_critique_dim=None, critique_confidence=0.0, slot_id='lunch',
        )
    assert np.allclose(np.asarray(out2.get('phase_a_posterior_mu', [0.0]*6)), np.zeros(6))
