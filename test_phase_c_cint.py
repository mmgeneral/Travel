"""Phase C4 tests: c_int anchoring diagnostics."""

import numpy as np
import pytest

from decision_engine import c_int_anchor_diagnostics
from config import C_INT, C_INT_GRID_FRACTIONS


def test_c_int_anchor_basic():
    samples = [
        [1.0, 0.5, 0.2],
        [0.8, 0.4, 0.1],
        [0.6, 0.3, 0.05],
        [0.7, 0.35, 0.15],
    ]
    diag = c_int_anchor_diagnostics(samples, current_c_int=C_INT,
                                     grid_fractions=C_INT_GRID_FRACTIONS)
    assert diag["median_gap"] > 0
    assert diag["q1_gap"] <= diag["median_gap"] <= diag["q3_gap"]
    assert diag["iqr_gap"] == pytest.approx(diag["q3_gap"] - diag["q1_gap"])
    assert diag["current_c_int"] == C_INT
    assert 0.0 <= diag["current_c_int_percentile"] <= 100.0
    assert len(diag["suggested_grid"]) == 3


def test_c_int_grid_uses_median_fraction():
    samples = [[1.0, 0.2]]
    diag = c_int_anchor_diagnostics(samples, current_c_int=0.05,
                                    grid_fractions=(0.1, 0.2))
    assert diag["suggested_grid"][0] == pytest.approx(0.08)
    assert diag["suggested_grid"][1] == pytest.approx(0.16)
