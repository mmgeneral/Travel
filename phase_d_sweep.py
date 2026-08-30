"""Phase D.4 sweep runner using real episode trajectories."""

from __future__ import annotations

import hashlib
import numpy as np
from typing import Iterable

import config
from synthetic_user import generate_trip_world
from experiment_arms import run_episode
import phase_d_metrics as metrics


def _aggregate(repeats: list[dict]) -> dict:
    """Return mean/std/min/max for numeric fields of a list of metric dicts."""
    if not repeats:
        return {}
    keys = [
        k
        for k in repeats[0].keys()
        if k not in ("arm", "rep", "final_mu", "final_sigma_diag")
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

    The synthetic world is generated once per (parameter, repetition).
    All arms run the same world, each with its own independent learner.
    No oracle values are used outside the synthetic-user response generator.
    """
    p_vals = list(p_crit_values) if p_crit_values is not None else list(config.P_CRIT_GRID)
    sig_vals = (
        list(sigma_item_values)
        if sigma_item_values is not None
        else list(config.SIGMA_ITEM_GRID)
    )
    c_vals = list(c_int_values) if c_int_values is not None else list(config.SWEEP_C_INT_GRID)
    n_repeats = repeats if repeats is not None else config.SWEEP_REPEATS

    all_raw: list[dict] = []
    all_aggregates: dict[tuple, dict] = {}
    calibration: dict = {}

    base_seed = 2026
    for p_crit in p_vals:
        for sigma_var in sig_vals:
            for c_int in c_vals:
                print(f"\n=== p_crit={p_crit} sigma_var={sigma_var} c_int={c_int} ===")
                arm_data = {arm: [] for arm in config.SWEEP_ARMS}

                for rep in range(n_repeats):
                    world_seed = base_seed * 1000 + int(p_crit * 100) + int(sigma_var * 100) + rep
                    rng_world = np.random.default_rng(world_seed)
                    world = generate_trip_world(
                        n_events=5,
                        n_candidates=8,
                        beta_star=None,
                        residual_multiplier=sigma_var,
                        rng=rng_world,
                        world_seed=world_seed,
                    )

                    for arm in config.SWEEP_ARMS:
                        # Stable seed independent of arm order.
                        arm_key = f"{world_seed}:{arm}"
                        arm_seed = int(hashlib.sha1(arm_key.encode()).hexdigest()[:8], 16)
                        rng_arm = np.random.default_rng(arm_seed)
                        st = run_episode(
                            world=world,
                            arm=arm,
                            p_crit=p_crit,
                            c_int=c_int,
                            rng=rng_arm,
                            force_ask=False,
                            evoi_mc_draws=200,
                        )

                        # ---- compute DV metrics from actual terminal state ----
                        rec = metrics.per_block_recovery(st.mu, world.beta_star)
                        cross = metrics.cross_block_cov_shrinkage(st.Sigma)

                        # contribution errors averaged over revision events
                        taste_errs: list[float] = []
                        context_errs: list[float] = []
                        eligible_true: list[bool] = []
                        eligible_correct: list[bool] = []
                        for ei, event in enumerate(world.events):
                            if ei >= len(st.proposal_trace):
                                break
                            x_sys = st.proposal_trace[ei]
                            y_true = event.true_best_index
                            delta_phi = event.phi[y_true] - event.phi[x_sys]
                            c_err = metrics.contribution_errors(delta_phi, st.mu, world.beta_star)
                            taste_errs.append(c_err["taste_error"])
                            context_errs.append(c_err["context_error"])

                            dom = metrics.dominant_block_accuracy(
                                delta_phi, st.mu, world.beta_star,
                                config.DOMINANT_BLOCK_DELTA,
                            )
                            if dom["eligible"]:
                                eligible_true.append(True)
                                eligible_correct.append(bool(dom["correct"]))

                        mean_taste_err = float(np.mean(taste_errs)) if taste_errs else 0.0
                        mean_ctx_err = float(np.mean(context_errs)) if context_errs else 0.0
                        eligible_count = len(eligible_true)
                        correct_count = sum(eligible_correct) if eligible_count else 0

                        burden = metrics.total_explicit_burden(
                            st.clarification_count, st.pairwise_count
                        )
                        mean_regret = float(np.mean(st.regret_trace)) if st.regret_trace else 0.0
                        cum_regret = float(np.sum(st.regret_trace))

                        row = {
                            "rep": rep,
                            "seed": world_seed,
                            "p_crit": p_crit,
                            "residual_multiplier": sigma_var,
                            "realized_sigma2_item": float(np.var(
                                np.concatenate([e.item_residual for e in world.events])
                            )),
                            "c_int": c_int,
                            "arm": arm,
                            "clarification_count": st.clarification_count,
                            "pairwise_count": st.pairwise_count,
                            "total_explicit_burden": burden["total_burden"],
                            "taste_recovery_error": rec["taste_norm"],
                            "context_recovery_error": rec["context_norm"],
                            "cross_block_cov_norm": cross["cross_block_cov_norm"],
                            "cross_block_cov_fraction": cross["cross_block_cov_fraction"],
                            "taste_contribution_error": mean_taste_err,
                            "context_contribution_error": mean_ctx_err,
                            "dominant_block_correct": correct_count,
                            "dominant_block_eligible_count": eligible_count,
                            "mean_regret": mean_regret,
                            "cumulative_regret": cum_regret,
                            "revision_count": st.revision_count,
                            "final_mu": st.mu.tolist(),
                            "final_sigma_diag": np.diag(st.Sigma).tolist(),
                        }
                        arm_data[arm].append(row)
                        all_raw.append(dict(row))

                for arm, rows in arm_data.items():
                    print(f"  Arm {arm} — raw:")
                    for row in rows:
                        print(
                            f"    rep {row['rep']}: "
                            f"clarify={row['clarification_count']} "
                            f"pairwise={row['pairwise_count']} "
                            f"regret={row['mean_regret']:.4f} "
                            f"rev={row['revision_count']} "
                            f"taste_err={row['taste_recovery_error']:.4f} "
                            f"ctx_err={row['context_recovery_error']:.4f}"
                        )
                    agg = _aggregate([{k: v for k, v in r.items() if k != "rep"} for r in rows])
                    key = (p_crit, sigma_var, c_int, arm)
                    print(f"    aggregate: {agg}")
                    all_aggregates[key] = agg

    # calibration based on top-2 S0 gaps from the first generated world.
    # (In a full implementation we would build dedicated calibration pools.)
    if n_repeats > 0 and p_vals and sig_vals:
        gap_list: list[float] = []
        for rep in range(n_repeats):
            seed_cal = base_seed * 1000 + 7 + rep
            world_seed_cal = seed_cal
            rng_cal = np.random.default_rng(world_seed_cal)
            world_cal = generate_trip_world(
                n_events=5,
                n_candidates=8,
                beta_star=None,
                residual_multiplier=0.0,
                rng=rng_cal,
                world_seed=world_seed_cal,
            )
            for ev in world_cal.events:
                s = np.sort(ev.s0_tilde)[::-1]
                if len(s) >= 2:
                    gap_list.append(float(s[0] - s[1]))
        if gap_list:
            arr = np.asarray(gap_list)
            median = float(np.median(arr))
            q1 = float(np.percentile(arr, 25))
            q3 = float(np.percentile(arr, 75))
            iqr = q3 - q1
            suggested = [round(median * f, 6) for f in (0.02, 0.10, 0.20)]
            current_c_int = 0.05
            percentile_rank = float(np.mean(arr <= current_c_int)) * 100.0
            calibration = {
                "median_gap": median,
                "q1_gap": q1,
                "q3_gap": q3,
                "iqr_gap": iqr,
                "current_c_int": current_c_int,
                "c_int_percentile": percentile_rank,
                "suggested_grid": suggested,
            }

    return {
        "raw_rows": all_raw,
        "aggregates": all_aggregates,
        "calibration": calibration,
    }
