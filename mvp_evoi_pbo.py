import itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
SLOTS = 6
OPTIONS_PER_SLOT = 4
ATTR_PER_OPT = 3          # [price, rating, is_outdoor]
D = SLOTS * ATTR_PER_OPT + 1   # 10 dims
POOL_SIZE = OPTIONS_PER_SLOT ** SLOTS   # 64
SIGMA0 = 1.0
N_MC = 200
C_INTERRUPTION = 0.05
T_ROUNDS = 30
N_REPS = 10
BASE_SEED = 12345
SPEARMAN_THRESHOLD = 0.95
# ----------------------------------------------------------------------


def sigmoid(x):
    """Numerically stable sigmoid for scalar / ndarray."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    out[~pos] = np.exp(x[~pos]) / (1.0 + np.exp(x[~pos]))
    return out


def generate_itineraries(seed):
    """Return itinerary feature matrix for SLOTS slots."""
    rng = np.random.default_rng(seed)
    slot_options = []
    for _ in range(SLOTS):
        opts = []
        for _ in range(OPTIONS_PER_SLOT):
            price = rng.uniform(50.0, 300.0)
            rating = rng.uniform(2.0, 5.0)
            is_outdoor = float(rng.integers(0, 2))
            opts.append(np.array([price, rating, is_outdoor], dtype=float))
        slot_options.append(np.array(opts))   # shape (4,3)

    phis = []
    for combo in itertools.product(range(OPTIONS_PER_SLOT), repeat=SLOTS):
        feats = [slot_options[s][combo[s]] for s in range(SLOTS)]
        feat = np.concatenate(feats)
        total_price = sum(f[0] for f in feats)
        feat = np.concatenate([feat, [total_price]])
        phis.append(feat)
    return np.array(phis, dtype=float)


def standardize_features(phi_vals):
    """Z-score standardize each feature dimension. Returns (phi_std, mean, std)."""
    mean = phi_vals.mean(axis=0)
    std = phi_vals.std(axis=0)
    std_safe = np.where(std < 1e-8, 1.0, std)
    phi_std = (phi_vals - mean) / std_safe
    return phi_std, mean, std_safe


def print_feature_scale_diagnostic(phi_vals):
    print("=== Feature scale diagnostic ===")
    for i in range(phi_vals.shape[1]):
        col = phi_vals[:, i]
        print(f"  dim {i:2d}: min={col.min():10.2f} max={col.max():10.2f} std={col.std():10.2f}")
    print()


def compute_metrics(mu, phi_vals, true_scores, f_true_best):
    """Return (spearman_corr, simple_regret)."""
    pred = phi_vals @ mu
    # handle all-equal predictions (mu = 0)
    if np.all(pred == pred[0]):
        # no useful ordering yet; recommend the first itinerary
        return 0.0, f_true_best - true_scores[0]
    corr, _ = spearmanr(pred, true_scores)
    if np.isnan(corr):
        corr = 0.0
    rec_idx = int(np.argmax(pred))
    regret = f_true_best - true_scores[rec_idx]
    return float(corr), float(regret)


def simulate_user(w_true, phi_vals, A, B, rng):
    """Return 1 if A chosen, 0 otherwise, under Bradley-Terry."""
    diff = phi_vals[A] - phi_vals[B]
    p = sigmoid(np.dot(w_true, diff))
    return 1 if rng.random() < p else 0


def neg_log_posterior(w, diffs, labels, sigma0):
    """Negative log-posterior for MAP Newton line search."""
    z = diffs @ w
    # log likelihood for each observation:
    #   y * log(sigmoid(z)) + (1-y) * log(1 - sigmoid(z))
    # using logaddexp to avoid overflow
    log_lik = np.sum(
        labels * (-np.logaddexp(0, -z)) + (1 - labels) * (-np.logaddexp(0, z))
    )
    log_prior = -0.5 * np.dot(w, w) / (sigma0 ** 2)
    return -(log_lik + log_prior)


def bayesian_fit(phi_vals, comparisons, sigma0=SIGMA0, max_iter=25, tol=1e-6):
    """MAP fit via manual Newton-Raphson + Laplace covariance."""
    if len(comparisons) == 0:
        d = phi_vals.shape[1]
        w = np.zeros(d)
        Sigma = np.eye(d) * (sigma0 ** 2)
        return w, Sigma

    diffs = []
    labels = []
    for a, b, y in comparisons:
        diffs.append(phi_vals[a] - phi_vals[b])
        labels.append(float(y))
    diffs = np.array(diffs)              # (N,d)
    labels = np.array(labels)            # (N,)
    d = diffs.shape[1]

    w = np.zeros(d)
    for _ in range(max_iter):
        z = diffs @ w
        p = sigmoid(z)
        grad = diffs.T @ (p - labels) + w / (sigma0 ** 2)
        # Hessian of negative log-posterior
        w_scale = p * (1.0 - p)
        H = (diffs * w_scale[:, None]).T @ diffs + np.eye(d) / (sigma0 ** 2)
        try:
            delta = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            H += np.eye(d) * 1e-6
            delta = np.linalg.solve(H, grad)

        # Backtracking line search: only accept a step that decreases
        # the negative log-posterior; otherwise shrink the step.
        current_nlp = neg_log_posterior(w, diffs, labels, sigma0)
        step_scale = 1.0
        max_backtrack = 20
        w_candidate = None
        for _ in range(max_backtrack):
            w_candidate = w - step_scale * delta
            candidate_nlp = neg_log_posterior(w_candidate, diffs, labels, sigma0)
            if candidate_nlp <= current_nlp:
                break
            step_scale *= 0.5
        else:
            # Backtracking exhausted without improvement; keep current w
            # and stop the outer loop to avoid a bad step.
            w_candidate = None
        if w_candidate is None:
            # No improving step found, stop iterating as requested
            break
        w = w_candidate
        effective_delta = step_scale * delta
        if np.linalg.norm(effective_delta) < tol:
            break

    # Laplace covariance at MAP
    z = diffs @ w
    p = sigmoid(z)
    w_scale = p * (1.0 - p)
    H = (diffs * w_scale[:, None]).T @ diffs + np.eye(d) / (sigma0 ** 2)
    Sigma = np.linalg.inv(H)
    return w, Sigma


def compute_evoi(mu, Sigma, phi_vals, comparisons, A, B, rng, c_int=C_INTERRUPTION, n_mc=N_MC):
    """EVOI(q) = P(A)*V(mu_A)+P(B)*V(mu_B)-V(mu)-c."""
    diff = phi_vals[A] - phi_vals[B]
    # current best under posterior mean
    V_mu = float((phi_vals @ mu).max())

    # Monte Carlo estimate of P(A)
    ws = rng.multivariate_normal(mu, Sigma, size=n_mc)
    prob_A = float(np.mean(sigmoid(ws @ diff)))

    # posterior mean after hypothetical outcome A
    comp_A = comparisons + [(A, B, 1)]
    mu_A, _ = bayesian_fit(phi_vals, comp_A)
    V_A = float((phi_vals @ mu_A).max())

    # posterior mean after hypothetical outcome B
    comp_B = comparisons + [(A, B, 0)]
    mu_B, _ = bayesian_fit(phi_vals, comp_B)
    V_B = float((phi_vals @ mu_B).max())

    return prob_A * V_A + (1.0 - prob_A) * V_B - V_mu - c_int


def select_query_thompson(mu, Sigma, phi_vals, rng):
    """Return (A,B) via two independent Thompson draws."""
    w1 = rng.multivariate_normal(mu, Sigma)
    scores1 = phi_vals @ w1
    A = int(np.argmax(scores1))
    # second independent draw, pick best other than A
    w2 = rng.multivariate_normal(mu, Sigma)
    scores2 = phi_vals @ w2
    scores2[A] = -np.inf
    B = int(np.argmax(scores2))
    return A, B


def rounds_to_threshold(spearman_curve, threshold=SPEARMAN_THRESHOLD):
    """Return the 1-indexed round at which spearman_curve first reaches
    `threshold` and stays at or above it for all subsequent rounds in
    this curve; return None if never reached."""
    hits = np.where(spearman_curve >= threshold)[0]
    if len(hits) == 0:
        return None
    # confirm it doesn't dip back below threshold afterward
    first_hit = hits[0]
    if np.all(spearman_curve[first_hit:] >= threshold):
        return first_hit + 1
    # if it dips back down, find the last stretch that stays above threshold
    for start in hits:
        if np.all(spearman_curve[start:] >= threshold):
            return start + 1
    return None


def run_condition(cond, phi_vals, w_true, seed, T=T_ROUNDS, c_int=C_INTERRUPTION):
    rng = np.random.default_rng(seed)
    comparisons = []
    mu = np.zeros(D)
    Sigma = np.eye(D) * (SIGMA0 ** 2)

    true_scores = phi_vals @ w_true
    f_true_best = float(true_scores.max())
    best_true_idx = int(np.argmax(true_scores))

    spearman_hist = np.zeros(T)
    regret_hist = np.zeros(T)
    questions = 0

    for t in range(T):
        if cond == "random_always_ask":
            A, B = rng.choice(np.arange(POOL_SIZE), size=2, replace=False)
        else:   # full_evoi or thompson_always_ask
            A, B = select_query_thompson(mu, Sigma, phi_vals, rng)

        if cond == "full_evoi":
            evoi = compute_evoi(mu, Sigma, phi_vals, comparisons, A, B, rng, c_int)
            if evoi <= 0:
                corr, regret = compute_metrics(mu, phi_vals, true_scores, f_true_best)
                spearman_hist[t] = corr
                regret_hist[t] = regret
                # no query, no refit
                continue

        # ask user
        y = simulate_user(w_true, phi_vals, A, B, rng)
        comparisons.append((A, B, y))
        mu, Sigma = bayesian_fit(phi_vals, comparisons)
        questions += 1

        corr, regret = compute_metrics(mu, phi_vals, true_scores, f_true_best)
        spearman_hist[t] = corr
        regret_hist[t] = regret

    return spearman_hist, regret_hist, questions


def main():
    # Fixed itinerary attributes across all repetitions/conditions
    phi_vals_raw = generate_itineraries(BASE_SEED)
    phi_vals, feat_mean, feat_std = standardize_features(phi_vals_raw)
    print_feature_scale_diagnostic(phi_vals)
    # Hidden true preference (same for every rep)
    rng_true = np.random.default_rng(BASE_SEED + 99)
    w_true = rng_true.normal(size=D)

    # EVOI magnitude diagnostic (early round)
    print("=== EVOI magnitude diagnostic (early round) ===")
    rng_diag = np.random.default_rng(12345)
    mu_diag = np.zeros(D)
    Sigma_diag = np.eye(D) * (SIGMA0 ** 2)
    comparisons_diag = []
    for i in range(3):
        A_d, B_d = select_query_thompson(mu_diag, Sigma_diag, phi_vals, rng_diag)
        for cc in [0.05, 0.01]:
            ev = compute_evoi(mu_diag, Sigma_diag, phi_vals, comparisons_diag, A_d, B_d, rng_diag, c_int=cc)
            print(f"  pair {i} (A={A_d},B={B_d}) c={cc:.2f}: EVOI={ev:.4f}")
    print()

    c_values = [0.05, 0.01]
    baseline_conds = ["thompson_always_ask", "random_always_ask"]

    # Run the two baselines once (they don't depend on c_interruption)
    baseline_results = {}
    for cond in baseline_conds:
        baseline_results[cond] = {
            "spearman": np.zeros((N_REPS, T_ROUNDS)),
            "regret": np.zeros((N_REPS, T_ROUNDS)),
            "questions": np.zeros(N_REPS),
        }
    for rep in range(N_REPS):
        rep_seed = BASE_SEED + 1000 + rep * 37
        for cond in baseline_conds:
            sp, rg, q = run_condition(cond, phi_vals, w_true, rep_seed)
            baseline_results[cond]["spearman"][rep] = sp
            baseline_results[cond]["regret"][rep] = rg
            baseline_results[cond]["questions"][rep] = q

    # Run full_evoi separately for each c_interruption
    full_evoi_by_c = {}
    for c_val in c_values:
        full_evoi_by_c[c_val] = {
            "spearman": np.zeros((N_REPS, T_ROUNDS)),
            "regret": np.zeros((N_REPS, T_ROUNDS)),
            "questions": np.zeros(N_REPS),
        }
        for rep in range(N_REPS):
            rep_seed = BASE_SEED + 1000 + rep * 37
            sp, rg, q = run_condition("full_evoi", phi_vals, w_true, rep_seed, c_int=c_val)
            full_evoi_by_c[c_val]["spearman"][rep] = sp
            full_evoi_by_c[c_val]["regret"][rep] = rg
            full_evoi_by_c[c_val]["questions"][rep] = q

    rounds = np.arange(1, T_ROUNDS + 1)

    def compute_stats(res):
        sp_mean = res["spearman"].mean(axis=0)
        sp_std = res["spearman"].std(axis=0, ddof=1)
        rg_mean = res["regret"].mean(axis=0)
        rg_std = res["regret"].std(axis=0, ddof=1)
        q_mean = res["questions"].mean()
        return sp_mean, sp_std, rg_mean, rg_std, q_mean

    baseline_stats = {}
    plot_series = []
    for cond in baseline_conds:
        s = compute_stats(baseline_results[cond])
        baseline_stats[cond] = s
        plot_series.append({
            "label": cond,
            "spearman_mean": s[0],
            "spearman_std": s[1],
            "regret_mean": s[2],
            "regret_std": s[3],
        })

    for c_val in c_values:
        print(f"=== c_interruption = {c_val} ===")
        # Build stats for full_evoi
        fe_stats = compute_stats(full_evoi_by_c[c_val])
        summary = {"full_evoi": fe_stats}
        for cond in baseline_conds:
            summary[cond] = baseline_stats[cond]

        # Rounds to reach Spearman >= threshold
        spearman_rounds = {}
        for cond in summary.keys():
            data = baseline_results[cond] if cond in baseline_conds else full_evoi_by_c[c_val]
            rep_rounds = []
            reached = 0
            for rep in range(N_REPS):
                r = rounds_to_threshold(data["spearman"][rep], SPEARMAN_THRESHOLD)
                if r is not None:
                    rep_rounds.append(r)
                    reached += 1
            mean_round = np.mean(rep_rounds) if rep_rounds else float('nan')
            spearman_rounds[cond] = (mean_round, reached)

        header = (f"{'Condition':<28}{'Final Spearman':<20}{'Final Regret':<20}"
                  f"{'Avg Questions':<15}{'Rounds to Sp>=0.95':<23}{'Reps reached':<15}")
        print(header)
        for cond in ["full_evoi", "thompson_always_ask", "random_always_ask"]:
            ms, ss, mr, sr, mq = summary[cond]
            mean_round, reached = spearman_rounds[cond]
            if np.isnan(mean_round):
                round_str = "N/A"
            else:
                round_str = f"{mean_round:.2f}"
            print(f"{cond:<28}{ms[-1]:<20.4f}{mr[-1]:<20.4f}{mq:<15.2f}"
                  f"{round_str:<23}{reached}/{N_REPS}")
        print()

        plot_series.append({
            "label": f"full_evoi (c={c_val})",
            "spearman_mean": fe_stats[0],
            "spearman_std": fe_stats[1],
            "regret_mean": fe_stats[2],
            "regret_std": fe_stats[3],
        })

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for series in plot_series:
        axes[0].plot(rounds, series["spearman_mean"], label=series["label"])
        axes[0].fill_between(rounds,
                             series["spearman_mean"] - series["spearman_std"],
                             series["spearman_mean"] + series["spearman_std"],
                             alpha=0.2)
        axes[1].plot(rounds, series["regret_mean"], label=series["label"])
        axes[1].fill_between(rounds,
                             series["regret_mean"] - series["regret_std"],
                             series["regret_mean"] + series["regret_std"],
                             alpha=0.2)
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("Mean Spearman correlation")
    axes[0].legend()
    axes[0].grid(True)
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("Mean Simple Regret")
    axes[1].legend()
    axes[1].grid(True)
    plt.tight_layout()
    plt.savefig("mvp_evoi_convergence_scaled.png", dpi=150)
    plt.close(fig)

    print()
    print("Note: Simple Regret saturates to 0 for all three conditions within the")
    print("first several rounds at this task scale (6 slots, 4096 itineraries) and")
    print("is not informative for distinguishing strategies beyond that point.")
    print("Spearman correlation and rounds-to-threshold are the more informative")
    print("metrics for this MVP's scale.")


if __name__ == "__main__":
    main()
