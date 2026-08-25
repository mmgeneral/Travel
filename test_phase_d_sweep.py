"""Phase D.4 sweep runner smoke test."""
import numpy as np
import pytest

import config
from phase_d_sweep import run_sweep


def test_sweep_returns_payload():
    res = run_sweep(
        p_crit_values=[0.0],
        sigma_item_values=[0.0],
        c_int_values=[0.05],
        repeats=3,
    )
    assert "raw_rows" in res
    assert "aggregates" in res
    assert "calibration" in res
    assert len(res["raw_rows"]) >= 3 * len(config.SWEEP_ARMS)
    assert "C0" in res["aggregates"]
