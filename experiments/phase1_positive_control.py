"""Phase‑1 exact matched pairwise positive‑control harness.

This script generates pairwise evidence from the learner's own model
(logistic choice, N(0,I) prior) and feeds it to the *existing*
:func:`likelihood.refit_laplace`.

Design:
  beta_star ~ N(0, I)
  Delta_phi ~ N(0, 0.25*I)   # numpy scale=0.5
  p_B = sigmoid(beta_star @ Delta_phi)
  win_B ~ Bernoulli(p_B)
  x_e = Delta_phi   if B wins
        -Delta_phi  if A wins

A and B are named before the outcome; the outcome determines which sign to
store, and *both* signs are kept.

The generator is independent from the learner – no S0, planner,
consideration set, natural revision, clarification, EVOI, or item residual.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from evidence import EvidenceRecord
from likelihood import refit_laplace


def _make_evidence(evidence_id: str, x_e: np.ndarray) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=evidence_id,
        thread_id="phase1-positive-control",
        ts="2024-01-01T00:00:00",
        event_type="replacement",
        learning=True,
        censored_feasibility=False,
        x_e=x_e.astype(float).tolist(),
        ask_eligible=False,
    )


def _generate_world(
    seed: int,
    n_records: int,
    *,
    tau: float = 0.5,
) -> tuple[np.ndarray, list[EvidenceRecord], list[np.ndarray], np.ndarray]:
    """Return (beta_star, records, deltas, unit_vector_u)."""
    seed_seq = np.random.SeedSequence(seed)
    beta_rng, design_rng, outcome_rng, diag_rng = [
        np.random.default_rng(s) for s in seed_seq.spawn(4)
    ]
    beta_star = beta_rng.normal(loc=0.0, scale=1.0, size=6)

    records: list[EvidenceRecord] = []
    deltas: list[np.ndarray] = []
    for i in range(n_records):
        delta = design_rng.normal(loc=0.0, scale=tau, size=6)
        deltas.append(delta)
        z = float(beta_star @ delta)
        p_win_B = 1.0 / (1.0 + np.exp(-z))
        win_B = outcome_rng.random() < p_win_B
        x_e = delta if win_B else -delta
        records.append(_make_evidence(f"obs-{i}", x_e))

    u = diag_rng.normal(size=6)
    u = u / (np.linalg.norm(u) + 1e-12)
    return beta_star, records, deltas, u


def _exact_grad(beta: np.ndarray, records: list[EvidenceRecord]) -> np.ndarray:
    """Closed‑form gradient for choice‑only objective."""
    beta = np.asarray(beta, dtype=float)
    grad = beta.copy()
    for rec in records:
        x = np.asarray(rec.x_e, dtype=float)
        z = float(beta @ x)
        grad -= (1.0 / (1.0 + np.exp(z))) * x
    return grad


def _exact_hess(beta: np.ndarray, records: list[EvidenceRecord]) -> np.ndarray:
    """Closed‑form Hessian for choice‑only objective."""
    beta = np.asarray(beta, dtype=float)
    H = np.eye(6)
    for rec in records:
        x = np.asarray(rec.x_e, dtype=float)
        z = float(beta @ x)
        s = 1.0 / (1.0 + np.exp(-z))
        H += s * (1.0 - s) * np.outer(x, x)
    return H


def _diagnose(
    records: list[EvidenceRecord],
    mu: np.ndarray,
    Sigma: np.ndarray,
    beta_star: np.ndarray,
    u: np.ndarray,
    deltas: list[np.ndarray],
) -> dict[str, float]:
    diff = mu - beta_star
    l2 = float(np.linalg.norm(diff))

    denom_cos = np.linalg.norm(mu) * np.linalg.norm(beta_star)
    cos = float(np.dot(mu, beta_star) / denom_cos) if denom_cos > 1e-12 else np.nan
    norm_ratio = float(np.linalg.norm(mu) / max(1e-12, np.linalg.norm(beta_star)))

    eig_sigma = np.linalg.eigvalsh(Sigma)
    max_eig_sigma = float(np.max(eig_sigma))

    H_exact = _exact_hess(mu, records)
    eig_negH = np.linalg.eigvalsh(H_exact)  # H is the *positive*-definite Hessian here
    min_eig_negH = float(np.min(eig_negH))

    grad_exact = _exact_grad(mu, records)
    grad_norm = float(np.linalg.norm(grad_exact))

    D = np.vstack(deltas)
    emp_cov = (D.T @ D) / D.shape[0]
    emp_eigs = np.linalg.eigvalsh(emp_cov)
    emp_eig_names = [f"emp_cov_eig_{i}" for i in range(6)]

    z_u = float((u @ diff) / max(1e-12, np.sqrt(u @ Sigma @ u)))

    row = {
        "l2_error": l2,
        "cosine": cos,
        "norm_ratio": norm_ratio,
        "max_eig_sigma": max_eig_sigma,
        "min_eig_negH": min_eig_negH,
        "grad_norm": grad_norm,
        **dict(zip(emp_eig_names, map(float, emp_eigs))),
        "z_u": z_u,
    }
    return row


def run_experiment(
    worlds: int,
    n_records: int,
    checkpoints: Iterable[int],
    *,
    seed: int = 12345,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    checkpoints = sorted(set(checkpoints))
    checkpoints = [c for c in checkpoints if c <= n_records]
    if not checkpoints:
        checkpoints = [n_records]

    raw_rows: list[dict] = []
    for world_idx in range(worlds):
        world_seed = seed * 100003 + world_idx
        beta_star, records, deltas, u = _generate_world(world_seed, n_records)

        for n in checkpoints:
            mu, Sigma = refit_laplace(records[:n])
            diag = _diagnose(records[:n], mu, Sigma, beta_star, u, deltas[:n])
            raw_rows.append({"world": world_idx, "n": n, **diag})

    raw_df = pd.DataFrame(raw_rows)

    agg_rows: list[dict] = []
    pooled_by_n: dict[int, list[float]] = {n: [] for n in checkpoints}
    for n in checkpoints:
        sub = raw_df[raw_df["n"] == n]
        pooled_by_n[n] = sub["z_u"].dropna().tolist()
        l2 = sub["l2_error"].dropna()
        q1, med, q3 = np.percentile(l2, [25, 50, 75]) if len(l2) else (np.nan, np.nan, np.nan)
        agg_rows.append(
            {
                "n": n,
                "median_l2": float(med),
                "iqr_l2": float(q3 - q1),
                "median_cos": float(np.median(sub["cosine"].dropna())),
                "median_norm_ratio": float(np.median(sub["norm_ratio"].dropna())),
                "median_max_eig_sigma": float(np.median(sub["max_eig_sigma"].dropna())),
                "median_min_eig_negH": float(np.median(sub["min_eig_negH"].dropna())),
                "median_grad_norm": float(np.median(sub["grad_norm"].dropna())),
            }
        )
    agg_df = pd.DataFrame(agg_rows)

    pooled_rows: list[dict] = []
    for n, arr in pooled_by_n.items():
        if not arr:
            continue
        a = np.asarray(arr)
        pooled_rows.append(
            {
                "n": n,
                "mean": float(np.mean(a)),
                "std": float(np.std(a)),
                "q05": float(np.percentile(a, 5)),
                "q25": float(np.percentile(a, 25)),
                "q50": float(np.percentile(a, 50)),
                "q75": float(np.percentile(a, 75)),
                "q95": float(np.percentile(a, 95)),
            }
        )
    pooled_df = pd.DataFrame(pooled_rows)

    return raw_df, agg_df, pooled_df


def run_smoke(
    worlds: int = 5,
    N: int = 100,
    checkpoints: tuple[int, ...] = (10, 30, 100),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return run_experiment(worlds, N, checkpoints, seed=2024)


def _write_outputs(raw: pd.DataFrame, agg: pd.DataFrame, pooled: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(out_dir / "raw_rows.csv", index=False)
    agg.to_csv(out_dir / "aggregate.csv", index=False)
    pooled.to_csv(out_dir / "pooled_z.csv", index=False)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Matched pairwise positive control")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--worlds", type=int)
    parser.add_argument("--N", type=int)
    parser.add_argument("--checkpoints", type=int, nargs="+")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/phase1_positive_control"))
    args = parser.parse_args(argv)

    if args.mode == "full":
        worlds = args.worlds or 200
        N = args.N or 1000
        cps = args.checkpoints or [10, 30, 100, 300, 1000]
    else:
        worlds = args.worlds or 10
        N = args.N or 100
        cps = args.checkpoints or [10, 30, 100]

    raw, agg, pooled = run_experiment(worlds, N, cps)
    _write_outputs(raw, agg, pooled, args.out_dir)

    print("== aggregate ==")
    print(agg.to_string(index=False))
    print("== pooled z_u ==")
    print(pooled.to_string(index=False))

    large = agg[agg["n"] >= 30].copy()
    if len(large) >= 2:
        fit = np.polyfit(np.log(large["n"]), np.log(large["median_l2"]), 1)
        print(f"\nlog-log slope (n>=30): {fit[0]:.3f}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
