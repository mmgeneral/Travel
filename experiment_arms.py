"""Phase D.2 — experiment arms and episode runner.

Implements the real multi-event trajectory used by the T1 study.
"""

from __future__ import annotations

import hashlib
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from config import LAMBDA, TAU, KAPPA
from evidence import EvidenceRecord
from likelihood import refit_laplace, prob_prompted
from preference_features import FEATURE_NAMES
from synthetic_user import SyntheticTripWorld
from decision_engine import generate_cross_block_questions, evaluate_gate


def _sample_from_prob(probs, rng):
    r = rng.random()
    for i, p in enumerate(np.cumsum(probs)):
        if r < p:
            return i
    return len(probs) - 1


def _synthetic_prompted_answer(beta_star, x_e, j_T, j_C, rng):
    probs = [prob_prompted(beta_star, x_e, j_T, j_C, o, lam=LAMBDA, tau=TAU, kappa=KAPPA)
             for o in range(3)]
    o = _sample_from_prob(np.asarray(probs), rng)
    if o == 0:
        return FEATURE_NAMES[j_T], o
    if o == 1:
        return FEATURE_NAMES[j_C], o
    return "other", o


def _compute_contender(event, mu, Sigma, score_sys, M=5, n_draws=200, rng=None):
    """Return (contender_indices, L_j) using the frozen Phase-B3 2·σ_pred rule."""
    top = np.argsort(score_sys)[::-1][:M]
    if len(top) == 0:
        return np.array([], dtype=int), np.zeros(6)
    best_idx = top[0]
    best_score = score_sys[best_idx]
    best_phi = event.phi[best_idx]
    contender = [best_idx]
    for idx in top[1:]:
        gap = best_score - score_sys[idx]
        dphi = event.phi[idx] - best_phi
        var = float(dphi @ Sigma @ dphi)
        var = max(0.0, var)
        sigma_pred = np.sqrt(var)
        if gap <= 2.0 * sigma_pred:
            contender.append(idx)
    contender = np.asarray(contender, dtype=int)
    phi_c = event.phi[contender]
    L_j = np.std(phi_c, axis=0)
    if L_j.shape[0] == 0:
        L_j = np.zeros(6)
    return contender, L_j


def _array_evoi_for_question(event, mu, Sigma, evidence_log, q, x_e, c_int, n_draws=100, rng=None):
    """EVSI for one cross-block question (frozen Phase-C definition).

    The decision value is deterministic: max_x (s0_tilde[x] + mu^T phi[x]).
    Monte-Carlo draws are used only to estimate answer probabilities.
    """
    if rng is None:
        rng = np.random.default_rng()
    j_T = int(q["j_T"])
    j_C = int(q["j_C"])
    opts = q["question_options"]

    mu_arr = np.asarray(mu, dtype=float).reshape(-1)
    Sigma_arr = np.asarray(Sigma, dtype=float)

    U_B = float(np.max(event.s0_tilde + event.phi @ mu_arr))

    beta_draws = rng.multivariate_normal(mu_arr, Sigma_arr, size=n_draws)

    p_o = np.zeros(3)
    for o in range(3):
        probs = np.array([
            prob_prompted(beta, x_e, j_T, j_C, o, lam=LAMBDA, tau=TAU, kappa=KAPPA)
            for beta in beta_draws
        ])
        p_o[o] = float(probs.mean())

    # Normalize for numerical safety
    p_sum = float(p_o.sum())
    if p_sum <= 0.0:
        p_o = np.ones(3) / 3.0
    else:
        p_o = p_o / p_sum

    U_Bo = np.zeros(3)
    for o in range(3):
        if p_o[o] <= 0.0:
            U_Bo[o] = U_B
            continue
        answer_option = "other" if o == 2 else opts[o]
        hyp_ev = EvidenceRecord(
            evidence_id="hyp_evsi",
            thread_id="hyp",
            ts="",
            event_type="clarification_answer",
            learning=True,
            censored_feasibility=False,
            x_e=x_e,
            question_options=opts,
            answer_option=answer_option,
            ask_eligible=True,
        ).model_dump()
        rows = list(evidence_log) + [hyp_ev]
        ev_rows = [EvidenceRecord(**r) if isinstance(r, dict) else r for r in rows]
        mu_o, _ = refit_laplace(ev_rows)
        U_Bo[o] = float(np.max(event.s0_tilde + event.phi @ mu_o))

    gross_evsi = float(np.dot(p_o, U_Bo)) - U_B
    net_evsi = gross_evsi - c_int
    return net_evsi, p_o.tolist()


@dataclass
class ArmState:
    arm: str
    evidence_log: list[dict] = field(default_factory=list)
    mu: np.ndarray = field(default_factory=lambda: np.zeros(6))
    Sigma: np.ndarray = field(default_factory=lambda: np.eye(6))
    clarification_count: int = 0
    pairwise_count: int = 0
    revision_count: int = 0
    proposal_trace: list[int] = field(default_factory=list)
    question_trace: list[dict] = field(default_factory=list)
    answer_trace: list[str] = field(default_factory=list)
    posterior_trace: list[dict] = field(default_factory=list)
    regret_trace: list[float] = field(default_factory=list)


def run_episode(
    *,
    world: SyntheticTripWorld,
    arm: str,
    p_crit: float = 0.0,
    c_int: float = 0.05,
    rng: Optional[np.random.Generator] = None,
    force_ask: bool = False,
    evoi_mc_draws: int = 100,
) -> ArmState:
    """Run one arm over the whole episode. Returns its final ArmState."""
    state = ArmState(arm=arm)
    if rng is None:
        rng = np.random.default_rng()

    for event_idx, event in enumerate(world.events):
        # ---- system choice depends on the arm's current mu ----
        score_sys = event.s0_tilde + event.phi @ state.mu
        x_sys = int(np.argmax(score_sys))
        state.proposal_trace.append(x_sys)

        # ---- true optimum ----
        u_true = event.s0_tilde + event.phi @ world.beta_star + event.item_residual
        y_true = int(np.argmax(u_true))

        if x_sys == y_true:
            state.regret_trace.append(0.0)
            continue

        # ---- natural explicit revision ----
        delta_phi = event.phi[y_true] - event.phi[x_sys]
        state.revision_count += 1

        if arm != "C4":
            rec = EvidenceRecord(
                evidence_id=f"repl_{event_idx}",
                thread_id="world",
                ts="",
                event_type="replacement",
                learning=True,
                censored_feasibility=False,
                x_e=delta_phi.tolist(),
                rejected_item=f"cand_{x_sys}",
                accepted_item=f"cand_{y_true}",
                ask_eligible=False,
                question_options=None,
                answer_option=None,
            ).model_dump()
            state.evidence_log.append(rec)

        # ---- spontaneous critique (same randomness across arms) ----
        crit_emitted = False
        crit_rng = np.random.default_rng(1000 * event_idx + 7)
        if p_crit > 0 and crit_rng.random() < p_crit:
            contrib = world.beta_star * delta_phi
            pos = contrib > 0
            if np.any(pos):
                j = int(np.argmax(np.where(pos, contrib, -np.inf)))
                if arm != "C4":
                    rec_crit = EvidenceRecord(
                        evidence_id=f"crit_{event_idx}",
                        thread_id="world",
                        ts="",
                        event_type="explicit_critique",
                        learning=True,
                        censored_feasibility=False,
                        x_e=(-event.phi[x_sys]).tolist(),
                        rejected_item=f"cand_{x_sys}",
                        accepted_item=None,
                        ask_eligible=False,
                        question_options=None,
                        answer_option=FEATURE_NAMES[j],
                        weight=1.0,
                    ).model_dump()
                    state.evidence_log.append(rec_crit)
                crit_emitted = True

        # ---- posterior update from any natural learning rows ----
        if arm != "C4":
            rows = [EvidenceRecord(**r) for r in state.evidence_log if r.get("learning")]
            if rows:
                state.mu, state.Sigma = refit_laplace(rows)

        # ---- regret ----
        state.regret_trace.append(float(u_true[y_true] - u_true[x_sys]))

        # ---- ask policy (C1,C2 only for clarification; C3 is pairwise) ----
        if arm in ("C0", "C4"):
            continue

        if arm == "C3":
            # C3 goes directly to pairwise block below
            pass
        else:
            # Use frozen Phase-C eligibility with real contender-based L_j
            contender, L_j = _compute_contender(
                event, state.mu, state.Sigma, score_sys,
                M=5, n_draws=evoi_mc_draws,
                rng=np.random.default_rng(1000 + event_idx),
            )
            if len(contender) <= 1:
                ask = False
                chosen_q = None
            else:
                q_candidates, eligible = generate_cross_block_questions(
                    Sigma=state.Sigma,
                    L_j=L_j.tolist(),
                    x_e=delta_phi.tolist(),
                )
                if not eligible or crit_emitted or not q_candidates:
                    ask = False
                    chosen_q = None
                else:
                    if arm == "C1":
                        chosen_q = q_candidates[0]
                        gate = evaluate_gate(
                            ask_eligible=True,
                            evoi_results=None,
                            asked_this_turn=False,
                            contender_size=int(len(contender)),
                            attribution_already_given=crit_emitted,
                            policy="always_ask",
                            force_ask=force_ask,
                        )
                        ask = gate.get("action") == "ask"
                    else:  # C2
                        if force_ask:
                            chosen_q = q_candidates[0]
                            gate = evaluate_gate(
                                ask_eligible=True,
                                evoi_results=None,
                                asked_this_turn=False,
                                contender_size=int(len(contender)),
                                attribution_already_given=crit_emitted,
                                policy="evoi_gated",
                                force_ask=True,
                            )
                            ask = gate.get("action") == "ask"
                        else:
                            evoi_list = []
                            for qi, q in enumerate(q_candidates):
                                net, p_o = _array_evoi_for_question(
                                    event,
                                    state.mu,
                                    state.Sigma,
                                    state.evidence_log,
                                    q,
                                    delta_phi.tolist(),
                                    c_int,
                                    n_draws=evoi_mc_draws,
                                    rng=np.random.default_rng(500 + event_idx * 10 + qi),
                                )
                                evoi_list.append({
                                    "question": q,
                                    "net_evoi": net,
                                    "p_o": p_o,
                                })
                            gate = evaluate_gate(
                                ask_eligible=True,
                                evoi_results=evoi_list,
                                asked_this_turn=False,
                                contender_size=int(len(contender)),
                                attribution_already_given=crit_emitted,
                                policy="evoi_gated",
                                force_ask=False,
                            )
                            ask = gate.get("action") == "ask"
                            if ask:
                                # pick question with best net EVOI
                                best = max(evoi_list, key=lambda x: x["net_evoi"])
                                chosen_q = best["question"]
                            else:
                                chosen_q = None

            if ask and chosen_q is not None:
                j_T = chosen_q["j_T"]
                j_C = chosen_q["j_C"]
                opts = chosen_q["question_options"]
                state.question_trace.append({
                    "j_T": j_T,
                    "j_C": j_C,
                    "question_options": opts,
                    "x_e": delta_phi.tolist(),
                })
                ans_rng = np.random.default_rng(1000 * event_idx + 3)
                answer_option, _ = _synthetic_prompted_answer(
                    world.beta_star, delta_phi.tolist(), j_T, j_C, ans_rng
                )
                state.answer_trace.append(answer_option)
                if arm != "C4":
                    rec_ans = EvidenceRecord(
                        evidence_id=f"ans_{event_idx}",
                        thread_id="world",
                        ts="",
                        event_type="clarification_answer",
                        learning=True,
                        censored_feasibility=False,
                        x_e=delta_phi.tolist(),
                        question_options=opts,
                        answer_option=answer_option,
                        ask_eligible=True,
                    ).model_dump()
                    state.evidence_log.append(rec_ans)
                    state.clarification_count += 1
                    rows = [EvidenceRecord(**r) for r in state.evidence_log if r.get("learning")]
                    state.mu, state.Sigma = refit_laplace(rows)

        # ---- C3 pairwise baseline (explicit item-vs-item query) ----
        if arm == "C3":
            order = np.argsort(score_sys)[::-1]
            a, b = int(order[0]), int(order[1])
            phi_a = event.phi[a]
            phi_b = event.phi[b]
            p_b_over_a = LAMBDA * 0.5 + (1.0 - LAMBDA) * (1.0 / (1.0 + np.exp(-world.beta_star.dot(phi_b - phi_a))))
            pair_rng = np.random.default_rng(1000 * event_idx + 5)
            if pair_rng.random() < p_b_over_a:
                rejected, accepted, x_e_pair = a, b, phi_b - phi_a
            else:
                rejected, accepted, x_e_pair = b, a, phi_a - phi_b
            rec_pair = EvidenceRecord(
                evidence_id=f"pair_{event_idx}",
                thread_id="world",
                ts="",
                event_type="replacement",
                learning=True,
                censored_feasibility=False,
                x_e=x_e_pair.tolist(),
                rejected_item=f"cand_{rejected}",
                accepted_item=f"cand_{accepted}",
                ask_eligible=False,
                question_options=None,
                answer_option=None,
            ).model_dump()
            state.evidence_log.append(rec_pair)
            state.pairwise_count += 1
            if arm != "C4":
                rows = [EvidenceRecord(**r) for r in state.evidence_log if r.get("learning")]
                state.mu, state.Sigma = refit_laplace(rows)

        state.posterior_trace.append({
            "mu": state.mu.tolist(),
            "Sigma_diag": np.diag(state.Sigma).tolist(),
            "proposal": x_sys,
            "true_best": y_true,
            "revision": x_sys != y_true,
        })

    state.posterior_trace.append({
        "mu": state.mu.tolist(),
        "Sigma_diag": np.diag(state.Sigma).tolist(),
    })
    return state
