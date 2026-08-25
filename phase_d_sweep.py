"""Phase D.4 sweep runner.

Runs the synthetic user generator over the specified axes
(p_crit, sigma_item_variance, c_int) and prints per-repetition raw
data for every experiment arm, followed by per-arm aggregates.
"""

from __future__ import annotations

import numpy as np
from typing import Iterable

import config
from synthetic_user import generate_synthetic_user
import phase_d_metrics as metrics


def _default_candidate_features() -> np.ndarray:
    """Return a small fixed candidate-feature matrix (6-D features)."""
    # Use identity rows so each candidate has one dominant feature.
    return np.eye(6)[:5]


def _compute_metrics_for_arm(
    *,
    delta_phi: np.ndarray,
    utilities: np.ndarray,
    chosen_index: int,
    beta_star: np.ndarray,
    mu_est: np.ndarray,
    arm: str,
) -> dict:
    """Compute DV metrics that depend on the arm only via explicit burden.

    The placeholder burden logic here is intentionally simple:
       - C0: never asks
       - C1: always asks one clarification
       - C2/C3/C4: currently treated as "no ask" until the full EVOI
         pipeline is plugged into this runner.
    This is sufficient to verify the sweep execution path.
    """
    if arm == "C0":
        clarify = 0
        pairwise = 0
    elif arm == "C1":
        clarify = 1
        pairwise = 0
    else:  # C2, C3, C4 placeholder (same as C0 until EVOI is integrated)
        clarify = 0
        pairwise = 0

    burden = metrics.total_explicit_burden(clarify, pairwise)
    block_err = metrics.contribution_errors(delta_phi, mu_est, beta_star)
    dom = metrics.dominant_block_accuracy(
        delta_phi, mu_est, beta_star, config.DOMINANT_BLOCK_DELTA
    )
    rec = metrics.per_block_recovery(mu_est, beta_star)
    reg = metrics.regret_from_utilities(utilities, chosen_index)

    return {
        "arm": arm,
        "burden_total": burden["total_burden"],
        "burden_clarification": burden["clarification_questions"],
        "burden_pairwise": burden["pairwise_extra"],
        "taste_norm": rec["taste_norm"],
        "context_norm": rec["context_norm"],
        "taste_error": block_err["taste_error"],
        "context_error": block_err["context_error"],
        "regret": reg,
        "dominant_block": dom,
        "chosen_index": int(chosen_index),
    }


def _aggregate(repeats: list[dict]) -> dict:
    """Return mean/std/min/max for numeric fields of a list of metric dicts."""
    if not repeats:
        return {}
    keys = [
        k
        for k in repeats[0].keys()
        if k not in ("arm", "dominant_block", "chosen_index")
    ]
    agg = {"arm": repeats[0]["arm"]}
    for k in keys:
        vals = [float(r[k]) for r in repeats if r[k] is not None]
        if vals:
            arr = np.asarray(vals)
            agg[f"{k}_mean"] = float(np.mean(arr))
            agg[f"{k}_std"] = float(np.std(arr))
            agg[f"{k}_min"] = float(np.min(arr))
            agg[f"{k}_max"] = float(np.max(arr))
    return agg


def run_sweep(
    *,
    p_crit_values: Iterable[float] | None = None,
    sigma_item_values: Iterable[float] | None = None,
    c_int_values: Iterable[float] | None = None,
    repeats: int | None = None,
) -> dict:
    """Perform the sweep and return raw_rows, aggregates and calibration.

    No oracle access: the synthetic world is generated once per
    (parameter, repetition) and all arms are run over the same world.
    Each arm's posterior mu must come from ``refit_laplace``.
    """
    p_vals = list(p_crit_values) if p_crit_values is not None else list(config.P_CRIT_GRID)
    sig_vals = (
        list(sigma_item_values)
        if sigma_item_values is not None
        else list(config.SIGMA_ITEM_GRID)
    )
    c_vals = list(c_int_values) if c_int_values is not None else list(config.SWEEP_C_INT_GRID)
    n_repeats = repeats if repeats is not None else config.SWEEP_REPEATS

    rng = np.random.default_rng(2026)

    all_raw: list[dict] = []
    all_aggregates: dict[str, dict] = {}
    calibration: dict = {}

    for p_crit in p_vals:
        for sigma_var in sig_vals:
            for c_int in c_vals:
                print(f"\n=== p_crit={p_crit} sigma_var={sigma_var} c_int={c_int} ===")
                features = _default_candidate_features()
                current_idx = 0

                arm_data = {arm: [] for arm in config.SWEEP_ARMS}

                for rep in range(n_repeats):
                    beta_star = rng.normal(size=6)
                    user = generate_synthetic_user(
                        features,
                        current_idx,
                        beta_star=beta_star,
                        sigma_item_variance=sigma_var,
                        p_crit=p_crit,
                        rng=rng,
                    )
                    utils = np.asarray(user["utilities"])
                    delta_phi = np.asarray(user["delta_phi"])
                    chosen = user["chosen_index"]
                    # The sweep runner without an experiment runner cannot
                    # refit Laplace; for the smoke test we still need per-arm
                    # mu.  Use zeros (prior) as placeholder — this is NOT
                    # used by any learner, only for aggregate printing.
                    mu_est = np.zeros(6)

                    for arm in config.SWEEP_ARMS:
                        m = _compute_metrics_for_arm(
                            delta_phi=delta_phi,
                            utilities=utils,
                            chosen_index=chosen,
                            beta_star=beta_star,
                            mu_est=mu_est,
                            arm=arm,
                        )
                        m["rep"] = rep
                        arm_data[arm].append(m)

                for arm, rows in arm_data.items():
                    print(f"  Arm {arm} — raw:")
                    for row in rows:
                        print(
                            f"    rep {row['rep']}: "
                            f"chosen={row['chosen_index']} "
                            f"taste_err={row['taste_error']:.4f} "
                            f"ctx_err={row['context_error']:.4f} "
                            f"regret={row['regret']:.4f} "
                            f"burden={row['burden_total']}"
                        )
                    agg = _aggregate([{k: v for k, v in r.items() if k != "rep"} for r in rows])
                    print(f"    aggregate: {agg}")
                    all_aggregates[arm] = agg

                for arm, rows in arm_data.items():
                    for r in rows:
                        all_raw.append(dict(r))

    # Minimal calibration (placeholder).  In the real runner this is computed
    # from the median top‑2 S_B gap of a calibration pool.
    calibration = {
        "median_gap": 0.0,
        "q1_gap": 0.0,
        "q3_gap": 0.0,
        "iqr_gap": 0.0,
        "current_c_int": 0.05,
        "suggested_grid": [0.02, 0.10, 0.20],
    }

    return {
        "raw_rows": all_raw,
        "aggregates": all_aggregates,
        "calibration": calibration,
    }
