import numpy as np
import time
from scipy.special import expit
from utils.metrics import *

def saga_admm(A, b, D, max_iter=1000, p_star=0.0,
              mu1=1e-3, mu2=1e-2, rho=1.0,
              step_size=0.01, batch_size=32):
    n, d = A.shape
    batch_size = min(batch_size, n)

    x = np.zeros(d)
    y = np.zeros(D.shape[0])
    lam_u = np.zeros(D.shape[0])

    # ---------------- 修正 1 & 2：正确的初始化 ----------------
    z_full = b * (A @ x)  # (n,) 修复了维度爆炸 Bug
    expit_z = expit(-z_full).reshape(-1, 1)  # (n, 1)

    # 严格保持 SAGA 的不变量：让 grad_table 的均值等于 avg_grad
    grad_table = -b[:, None] * A * expit_z  # (n, d) 存储所有样本的初始梯度
    avg_grad = np.mean(grad_table, axis=0)  # (d,)

    gaps, primals, duals, time_list = [], [], [], []
    start_time = time.time()

    for k in range(max_iter):
        # 1. 随机采样
        indices = np.random.choice(n, size=batch_size, replace=False)

        # 2. 计算当前 batch 真实梯度
        z = b[indices] * (A[indices] @ x)  # (batch_size,)
        current_grad_batch = -b[indices, None] * A[indices] * expit(-z[:, None])  # (batch_size, d)

        # 3. 构造 SAGA 无偏估计器 (这部分你的数学推导完全正确！)
        saga_grad_batch = current_grad_batch - grad_table[indices] + avg_grad[None, :]  # (batch_size, d)
        saga_grad_mean = np.mean(saga_grad_batch, axis=0)  # (d,)

        # ---------------- 修正 3：向量化更新梯度表 ----------------
        grad_old_batch = grad_table[indices]
        grad_table[indices] = current_grad_batch
        # 批量更新均值: avg_new = avg_old + sum(新梯度 - 老梯度) / n
        avg_grad = avg_grad + np.sum(current_grad_batch - grad_old_batch, axis=0) / n

        # 4. ADMM y 更新
        u = D @ x + lam_u
        y = np.sign(u) * np.maximum(np.abs(u) - mu2 / rho, 0)

        # 5. ADMM x 更新 (包含 L2 正则化)
        saga_est = saga_grad_mean + mu1 * x
        x = x - step_size * (saga_est + rho * D.T @ (D @ x - y + lam_u))

        # 6. ADMM 对偶变量更新
        lam_u_prev = lam_u.copy()
        lam_u = lam_u + D @ x - y

        # 记录指标
        gap = objective_gap(x, y, D, A, b, mu1, mu2, p_star)
        pr = primal_residual(D, x, y)
        dr = dual_residual(lam_u_prev, lam_u, rho, D)

        gaps.append(gap)
        primals.append(pr)
        duals.append(dr)
        time_list.append(time.time() - start_time)

    return {"gap": gaps, "primal": primals, "dual": duals, "time": time_list}