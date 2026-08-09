import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
SLOTS = 3
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
    """Return phi-matrix of shape (64,10)."""
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
    for i in range(OPTIONS_PER_SLOT):
        for j in range(OPTIONS_PER_SLOT):
            for k in range(OPTIONS_PER_SLOT):
                feat = np.concatenate([
                    slot_options[0][i],
                    slot_options[1][j],
                    slot_options[2][k],
                ])
                total_price = slot_options[0][i][0] + slot_options[1][j][0] + slot_options[2][k][0]
                feat = np.concatenate([feat, [total_price]])
                phis.append(feat)
    return np.array(phis, dtype=float)


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
        w = w - delta
        if np.linalg.norm(delta) < tol:
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
    phi_vals = generate_itineraries(BASE_SEED)
    # Hidden true preference (same for every rep)
    rng_true = np.random.default_rng(BASE_SEED + 99)
    w_true = rng_true.normal(size=D)

    conditions = ["full_evoi", "thompson_always_ask", "random_always_ask"]
    results = {c: {"spearman": np.zeros((N_REPS, T_ROUNDS)),
                   "regret": np.zeros((N_REPS, T_ROUNDS)),
                   "questions": np.zeros(N_REPS)}
               for c in conditions}

    for rep in range(N_REPS):
        rep_seed = BASE_SEED + 1000 + rep * 37
        for cond in conditions:
            sp, rg, q = run_condition(cond, phi_vals, w_true, rep_seed)
            results[cond]["spearman"][rep] = sp
            results[cond]["regret"][rep] = rg
            results[cond]["questions"][rep] = q

    # Aggregate stats
    summary = {}
    rounds = np.arange(1, T_ROUNDS + 1)
    for cond in conditions:
        mean_sp = results[cond]["spearman"].mean(axis=0)
        std_sp = results[cond]["spearman"].std(axis=0, ddof=1)
        mean_rg = results[cond]["regret"].mean(axis=0)
        std_rg = results[cond]["regret"].std(axis=0, ddof=1)
        mean_q = results[cond]["questions"].mean()
        summary[cond] = (mean_sp, std_sp, mean_rg, std_rg, mean_q)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for cond in conditions:
        ms, ss, mr, sr, _ = summary[cond]
        axes[0].plot(rounds, ms, label=cond)
        axes[0].fill_between(rounds, ms - ss, ms + ss, alpha=0.2)
        axes[1].plot(rounds, mr, label=cond)
        axes[1].fill_between(rounds, mr - sr, mr + sr, alpha=0.2)
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("Mean Spearman correlation")
    axes[0].legend()
    axes[0].grid(True)
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("Mean Simple Regret")
    axes[1].legend()
    axes[1].grid(True)
    plt.tight_layout()
    plt.savefig("mvp_evoi_convergence.png", dpi=150)
    plt.close(fig)

    # Print summary table
    header = f"{'Condition':<28}{'Final Spearman':<20}{'Final Regret':<20}{'Avg Questions':<15}"
    print(header)
    for cond in conditions:
        ms, ss, mr, sr, mq = summary[cond]
        print(f"{cond:<28}{ms[-1]:<20.4f}{mr[-1]:<20.4f}{mq:<15.2f}")


if __name__ == "__main__":
    main()
