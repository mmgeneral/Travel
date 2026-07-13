"""
Preferential BO 最小驗證 demo

目的：驗證 BT model + GP surrogate + Thompson Sampling 這條演算法邏輯
是否真的能從 pairwise comparison 裡收斂到正確的偏好權重。

做法：
  1. 合成一組「假裝知道的」真實偏好權重 w_true（僅供驗證用，
     真實系統裡永遠不知道這個值）
  2. 建立一批候選行程，每個行程有 d 個屬性 phi(S)
  3. 每輪：Thompson Sampling 選一對候選 (A, B)
          模擬使用者根據 w_true 回答「A 還是 B」（BT model 抽樣）
          用貝葉斯線性迴歸更新對 w 的後驗估計
  4. 印出每輪：
     - 後驗對 w 的不確定性（協方差矩陣的 trace，越小代表越確定）
     - 目前估計 w_hat 跟真實 w_true 的誤差（cosine similarity）
     - 目前後驗對候選解排序，跟用 w_true 排序的 Spearman correlation

跑法：
    /home/mmg/venv/bin/python pbo_demo.py
"""
import numpy as np
from scipy.stats import spearmanr

np.random.seed(42)

# ------------------------------------------------------------
# 1. 合成情境
# ------------------------------------------------------------

D = 4          # 屬性維度：例如 [價格(負向已轉正), 等待時間(負向已轉正), 素食選項, 中文菜單]
N_CANDIDATES = 30   # 候選行程池大小
N_ROUNDS = 20       # 最多問幾輪

# 真實使用者偏好權重（僅供驗證，正式系統裡永遠不知道這個值）
w_true = np.array([0.5, 0.3, 0.1, 0.1])
w_true = w_true / np.linalg.norm(w_true)

print("=" * 70)
print("Preferential BO 最小驗證 Demo")
print("=" * 70)
print(f"\n屬性維度 D = {D}（[價格滿意度, 等待時間滿意度, 素食選項, 中文菜單]）")
print(f"候選行程池大小 = {N_CANDIDATES}")
print(f"(驗證用) 真實偏好權重 w_true = {np.round(w_true, 3)}")

# 候選行程的屬性向量，每個維度介於 0~1
candidates = np.random.uniform(0, 1, size=(N_CANDIDATES, D))


def true_utility(phi):
    """f(S) = <w_true, phi(S)>，僅供模擬使用者回答用。"""
    return phi @ w_true


def simulate_user_choice(phi_a, phi_b, noise=0.3):
    """BT model：P(A > B) = sigmoid(f(A) - f(B))，加一點雜訊模擬人類不完全理性。"""
    diff = true_utility(phi_a) - true_utility(phi_b)
    p_a_wins = 1.0 / (1.0 + np.exp(-diff / noise))
    return 1 if np.random.rand() < p_a_wins else -1  # 1 = A 勝, -1 = B 勝


# ------------------------------------------------------------
# 2. 貝葉斯線性迴歸：對 w 的後驗估計
#    （用線性效用假設 f(S) = <w, phi(S)> 簡化 GP，方便手動實作驗證）
# ------------------------------------------------------------

class BayesianPreferenceModel:
    """
    對 w 的高斯後驗估計，用 Laplace 近似逐輪更新
    （標準做法：從 BT model 的 log-likelihood 做二階泰勒展開）
    """

    def __init__(self, dim, prior_var=1.0):
        self.dim = dim
        self.mean = np.zeros(dim)             # 後驗均值 (w 的估計)
        self.cov = np.eye(dim) * prior_var    # 後驗協方差 (不確定性)

    def update(self, phi_a, phi_b, choice, n_newton_steps=3):
        """
        用 Newton-Raphson 對這一輪的 log-likelihood 做正確的 Laplace 近似更新，
        比單步梯度上升穩定，避免學習率選不好造成的震盪。
        choice: 1 表示 A 勝，-1 表示 B 勝
        """
        diff = phi_a - phi_b
        prior_mean, prior_prec = self.mean.copy(), np.linalg.inv(self.cov)

        # 從 prior 出發，做幾步 Newton 更新去逼近這一輪觀測後的 MAP
        w = self.mean.copy()
        for _ in range(n_newton_steps):
            z = choice * (w @ diff)
            sigmoid = 1.0 / (1.0 + np.exp(-z))
            grad = prior_prec @ (prior_mean - w) + choice * diff * (1 - sigmoid)
            hess = -prior_prec - np.outer(diff, diff) * sigmoid * (1 - sigmoid)
            w = w - np.linalg.solve(hess, grad)

        self.mean = w
        z = choice * (w @ diff)
        sigmoid = 1.0 / (1.0 + np.exp(-z))
        info = np.outer(diff, diff) * sigmoid * (1 - sigmoid)
        self.cov = np.linalg.inv(prior_prec + info)

    def sample(self):
        """Thompson Sampling: 從後驗採樣一個 w_hat。"""
        return np.random.multivariate_normal(self.mean, self.cov)

    def uncertainty(self):
        """協方差矩陣的 trace，越小代表對 w 越有把握。"""
        return np.trace(self.cov)


# ------------------------------------------------------------
# 3. 主迴圈：Thompson Sampling + 更新 + 記錄
# ------------------------------------------------------------

model = BayesianPreferenceModel(dim=D)

print("\n" + "-" * 70)
print(f"{'輪次':<6}{'不確定性(trace)':<18}{'w_hat vs w_true (cos sim)':<28}{'排序相關性':<12}")
print("-" * 70)

for round_idx in range(1, N_ROUNDS + 1):
    # Thompson Sampling: 採樣兩次，各自取候選池中的 argmax，組成一對
    w_hat_1 = model.sample()
    w_hat_2 = model.sample()
    scores_1 = candidates @ w_hat_1
    scores_2 = candidates @ w_hat_2
    idx_a = np.argmax(scores_1)
    idx_b = np.argmax(scores_2)

    # 避免選到同一個候選（步長至少要有意義）
    if idx_a == idx_b:
        idx_b = np.argsort(scores_2)[-2]

    phi_a, phi_b = candidates[idx_a], candidates[idx_b]

    # 模擬使用者回答
    choice = simulate_user_choice(phi_a, phi_b)

    # 更新後驗
    model.update(phi_a, phi_b, choice)

    # 記錄診斷指標
    unc = model.uncertainty()
    w_hat_norm = model.mean / (np.linalg.norm(model.mean) + 1e-8)
    cos_sim = np.dot(w_hat_norm, w_true)

    true_scores = candidates @ w_true
    est_scores = candidates @ model.mean
    rank_corr, _ = spearmanr(true_scores, est_scores)

    winner = "A" if choice == 1 else "B"
    print(f"{round_idx:<6}{unc:<18.4f}{cos_sim:<28.4f}{rank_corr:<12.4f}"
          f"  (第{round_idx}輪: 候選#{idx_a} vs #{idx_b}, 使用者選 {winner})")

print("-" * 70)
print(f"\n最終 w_hat = {np.round(model.mean / np.linalg.norm(model.mean), 3)}")
print(f"真實 w_true = {np.round(w_true, 3)}")
print(f"最終 cosine similarity = {cos_sim:.4f}")
print(f"最終排序相關性 (Spearman) = {rank_corr:.4f}")

print("\n" + "=" * 70)
print("觀察重點：")
print("  1. 不確定性 (trace) 應隨輪次單調下降")
print("  2. cosine similarity 應隨輪次上升，趨近 1.0")
print("  3. 排序相關性應隨輪次上升，趨近 1.0")
print("=" * 70)