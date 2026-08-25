"""Phase D.2 — experiment arms and episode runner.

Implements the real multi-event trajectory used by the T1 study.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from config import LAMBDA, TAU, KAPPA
from evidence import EvidenceRecord
from likelihood import refit_laplace, prob_prompted
from preference_features import FEATURE_NAMES
from synthetic_user import SyntheticTripWorld


def _make_question(x_e):
    """Return (j_T, j_C, question_options) using the current event's x_e."""
    x = np.asarray(x_e, dtype=float)
    j_T = int(np.argmax(np.abs(x[:3])))
    j_C = 3 + int(np.argmax(np.abs(x[3:])))
    opts = [FEATURE_NAMES[j_T], FEATURE_NAMES[j_C], "other"]
    return j_T, j_C, opts


def _expected_max_utility(event, mu, Sigma, n_draws=200, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    draws = rng.multivariate_normal(mu, Sigma, size=n_draws)
    utils = event.s0_tilde[None, :] + draws @ event.phi.T
    return float(utils.max(axis=1).mean())


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
    evoi_mc_draws: int = 200,
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
                evidence_id=f"repl_{arm}_{event_idx}",
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
                attribution_already_given=False,
            ).model_dump()
            state.evidence_log.append(rec)

        # ---- spontaneous critique (same randomness across arms) ----
        crit_rng = np.random.default_rng(1000 * event_idx + 7)
        if p_crit > 0 and crit_rng.random() < p_crit:
            contrib = world.beta_star * delta_phi
            pos = contrib > 0
            if np.any(pos):
                j = int(np.argmax(np.where(pos, contrib, -np.inf)))
                if arm != "C4":
                    rec_crit = EvidenceRecord(
                        evidence_id=f"crit_{arm}_{event_idx}",
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
                        attribution_already_given=True,
                    ).model_dump()
                    state.evidence_log.append(rec_crit)

        # ---- posterior update from any natural learning rows ----
        if arm != "C4":
            rows = [EvidenceRecord(**r) for r in state.evidence_log if r.get("learning")]
            if rows:
                state.mu, state.Sigma = refit_laplace(rows)

        # ---- regret ----
        state.regret_trace.append(float(u_true[y_true] - u_true[x_sys]))

        # ---- ask policy ----
        if arm in ("C0", "C4"):
            continue

        j_T, j_C, opts = _make_question(delta_phi)
        ask_eligible = True
        q = {"j_T": j_T, "j_C": j_C, "question_options": opts, "x_e": delta_phi.tolist()}
        state.question_trace.append(q)

        net_evoi = 0.0
        if arm == "C2":
            hyp_probs = [prob_prompted(world.beta_star, delta_phi.tolist(), j_T, j_C, o,
                                       lam=LAMBDA, tau=TAU, kappa=KAPPA)
                         for o in range(3)]
            gross_evoi = 0.0
            for o in range(3):
                answer_option = "other" if o == 2 else opts[o]
                new_rows = list(state.evidence_log)
                new_rows.append(EvidenceRecord(
                    evidence_id=f"hyp_evoi_{event_idx}_{o}",
                    thread_id="hyp",
                    ts="",
                    event_type="clarification_answer",
                    learning=True,
                    censored_feasibility=False,
                    x_e=delta_phi.tolist(),
                    question_options=opts,
                    answer_option=answer_option,
                    ask_eligible=True,
                ).model_dump())
                rows = [EvidenceRecord(**r) for r in new_rows if r.get("learning")]
                mu_h, Sig_h = refit_laplace(rows)
                hyp_ev = _expected_max_utility(event, mu_h, Sig_h, n_draws=evoi_mc_draws, rng=rng)
                gross_evoi += hyp_probs[o] * hyp_ev
            cur_ev = _expected_max_utility(event, state.mu, state.Sigma, n_draws=evoi_mc_draws, rng=rng)
            net_evoi = gross_evoi - cur_ev - c_int

        if (arm == "C1" and ask_eligible) or (arm == "C2" and net_evoi > 0) or force_ask:
            ans_rng = np.random.default_rng(1000 * event_idx + 3)
            answer_option, _ = _synthetic_prompted_answer(world.beta_star, delta_phi.tolist(), j_T, j_C, ans_rng)
            state.answer_trace.append(answer_option)
            if arm != "C4":
                rec_ans = EvidenceRecord(
                    evidence_id=f"ans_{arm}_{event_idx}",
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
                evidence_id=f"pair_{arm}_{event_idx}",
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
                attribution_already_given=False,
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
