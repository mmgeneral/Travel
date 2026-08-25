"""Phase C1 tests: cross-block question candidate generation."""
import numpy as np
import pytest

from decision_engine import generate_cross_block_questions


def _iv() -> tuple[np.ndarray, list[float]]:
    Sigma = np.eye(6)
    L_j = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    return Sigma, L_j


def test_pure_taste_replacement_no_questions():
    Sigma, L_j = _iv()
    x_e = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # only cuisine_match (taste)
    questions, ask = generate_cross_block_questions(Sigma=Sigma, L_j=L_j, x_e=x_e)
    assert questions == []
    assert ask is False


def test_pure_context_replacement_no_questions():
    Sigma, L_j = _iv()
    x_e = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]  # only travel_min (context)
    questions, ask = generate_cross_block_questions(Sigma=Sigma, L_j=L_j, x_e=x_e)
    assert questions == []
    assert ask is False


def test_cross_block_questions_generated_with_order():
    Sigma = np.eye(6)
    L_j = [1.0, 2.0, 1.0, 3.0, 1.0, 1.0]  # amplify fame (1) and travel (3)
    x_e = [0.5, 0.8, 0.0, 0.5, 0.0, 0.0]
    questions, ask = generate_cross_block_questions(Sigma=Sigma, L_j=L_j, x_e=x_e)
    assert ask is True
    assert len(questions) >= 1
    for q in questions:
        assert q["j_T"] in [0, 1, 2]
        assert q["j_C"] in [3, 4, 5]
        assert q["question_options"][2] == "other"
    # The best pair should use j_T=1 (fame) and j_C=3 (travel)
    assert questions[0]["j_T"] == 1
    assert questions[0]["j_C"] == 3
