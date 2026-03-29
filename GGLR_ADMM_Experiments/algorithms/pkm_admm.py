import numpy as np
import time
from utils.metrics import objective_gap, primal_residual, dual_residual
from scipy.special import expit


def pkm_admm(A, b, D, max_iter=1000, p_star=0.0,
             mu1=1e-3, mu2=1e-2, rho=1.0,
             step_size=0.01, batch_size=32,
             gamma=1.0, tau=0.5, varrho=0.3, update_prob=0.1):
    """
    PKM-ADMM (Proximal Katyusha Momentum ADMM) 求解 GGLR 问题
    :param mu1: L2 正则化系数 (对应公式中的 mu1)
    :param mu2: L1 图正则系数 (对应公式中的 mu2/lam)
    :param gamma: Inexact step 参数
    :param tau: Katyusha 动量参数 1
    :param varrho: Katyusha 动量参数 2
    :param update_prob: 快照更新概率 p_t
    """
    n, d = A.shape
    n_edges = D.shape[0]
    batch_size = min(batch_size, n)

    # 1. 初始化变量
    # x: 辅助变量 (用于计算梯度), z: 核心变量, q: 动量变量, w: 快照变量
    z = np.zeros(d)
    w = np.zeros(d)
    q = np.zeros(d)
    x = np.zeros(d)

    y = np.zeros(n_edges)
    lam_u = np.zeros(n_edges)  # 对偶变量

    # 初始快照全梯度计算
    def get_full_grad(theta):
        logits = b * (A @ theta)
        coef = -b * expit(-logits)
        return (A.T @ coef) / n + mu1 * theta

    full_grad_w = get_full_grad(w)

    gaps, primals, duals, time_list = [], [], [], []
    start_time = time.time()

    for t in range(max_iter):
        # --- Step 1: y 更新 (Soft-thresholding) ---
        u_t = D @ z + lam_u
        y = np.sign(u_t) * np.maximum(np.abs(u_t) - mu2 / rho, 0)

        # --- Step 2: Katyusha Momentum (x_{t+1}) ---
        # x_next = tau * z + varrho * w + (1 - tau - varrho) * q
        x = tau * z + varrho * w + (1 - tau - varrho) * q

        # --- Step 3: SVRG Gradient (v_{t+1}) ---
        idx = np.random.choice(n, size=batch_size, replace=False)
        A_batch = A[idx]
        b_batch = b[idx]

        # 当前 x 的随机梯度
        logits_x = b_batch * (A_batch @ x)
        grad_x_batch = (A_batch.T @ (-b_batch * expit(-logits_x))) / batch_size + mu1 * x

        # 快照 w 的随机梯度
        logits_w = b_batch * (A_batch @ w)
        grad_w_batch = (A_batch.T @ (-b_batch * expit(-logits_w))) / batch_size + mu1 * w

        # 方差缩减梯度
        v = grad_x_batch - grad_w_batch + full_grad_w

        # --- Step 4: Inexact Step (z_{t+1}) ---
        # z_next = z - (eta/gamma) * [v + rho * D.T @ (D@z - y + lam_u)]
        z_prev = z.copy()
        grad_admm = v + rho * D.T @ (D @ z - y + lam_u)
        z = z - (step_size / gamma) * grad_admm

        # --- Step 5: q 变量更新 ---
        q_next = x + tau * (z - z_prev)
        q = q_next  # 更新 q 用于下一轮

        # --- Step 6: 对偶变量更新 ---
        lam_u_prev = lam_u.copy()
        lam_u = lam_u + D @ z - y

        # --- Step 7: 快照更新 (w_{t+1}) ---
        if np.random.rand() < update_prob:
            w = q.copy()  # 根据算法，这里通常取上一个 q
            full_grad_w = get_full_grad(w)

        # --- 指标记录 ---
        # 使用 z 作为最终解的输出
        gap = objective_gap(z, y, D, A, b, mu1, mu2, p_star)
        pr = primal_residual(D, z, y)
        dr = dual_residual(lam_u_prev, lam_u, rho, D)

        gaps.append(gap)
        primals.append(pr)
        duals.append(dr)
        time_list.append(time.time() - start_time)

    return {"gap": gaps, "primal": primals, "dual": duals, "time": time_list}