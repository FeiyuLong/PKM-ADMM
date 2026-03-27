import numpy as np
import time
from utils.metrics import *
from scipy.special import expit

def saga_admm(A, b, D, max_iter=1000, p_star=0.0,
              mu = 1e-3, lam = 1e-2, rho = 1.0,
              step_size=0.01, batch_size=32):  # 新增batch_size参数，默认32
    """
    SAGA-ADMM求解GGLR问题，添加步长和batch_size参数
    :param A: 样本特征矩阵 (n_samples, n_features)
    :param b: 标签向量 (n_samples,)
    :param D: 图关联矩阵 (n_edges, n_features)
    :param mu: L2正则化系数
    :param lam: L1正则化系数
    :param rho: ADMM惩罚系数
    :param max_iter: 最大迭代次数
    :param step_size: 梯度下降步长
    :param p_star: GGLR问题的真实最优值
    :param batch_size: 随机采样的批次大小，默认32
    :return: 收敛指标（gap, primal, dual）
    """
    n, d = A.shape
    batch_size = min(batch_size, n)  # 确保batch_size不超过样本数
    x = np.zeros(d)
    y = np.zeros(D.shape[0])
    lam_u = np.zeros(D.shape[0])
    grad_table = np.zeros((n, d))  # 存储每个样本的历史梯度
    # 初始全梯度（向量操作）
    z_full = b[:, None] * A @ x  # (n, 1)
    # avg_grad = np.mean(-b[:, None] * A * expit(-z_full), axis=0)  # (d,)
    expit_z = expit(-z_full).reshape(-1, 1)  # 果z_full是样本维度 (500,) → 重塑为 (500, 1)
    avg_grad = np.mean(-b[:, None] * A * expit_z, axis=0)  # (d,)
    gaps, primals, duals = [], [], []
    time_list = []
    start_time = time.time()

    for k in range(max_iter):
        # 随机采样batch_size个样本
        indices = np.random.choice(n, size=batch_size, replace=False)

        # 批量计算当前梯度，向量操作
        z = b[indices] * (A[indices] @ x)  # (batch_size,)
        current_grad_batch = -b[indices, None] * A[indices] * expit(-z[:, None])  # (batch_size, d)
        current_grad_mean = np.mean(current_grad_batch, axis=0)  # (d,)

        # SAGA方差缩减梯度计算
        saga_grad_batch = current_grad_batch - grad_table[indices] + avg_grad[None, :]  # (batch_size, d)
        saga_grad_mean = np.mean(saga_grad_batch, axis=0)  # (d,)

        # 更新梯度表和平均梯度
        for i, idx in enumerate(indices):
            grad_old = grad_table[idx].copy()
            grad_new = current_grad_batch[i]  # 直接用batch位置索引，正确对应样本
            grad_table[idx] = grad_new
            avg_grad = avg_grad - (grad_old - grad_new) / n  # 数学公式与原代码完全一致

        # y更新：软阈值操作
        u = D @ x + lam_u
        y = np.sign(u) * np.maximum(np.abs(u) - lam / rho, 0)

        # x更新：SAGA梯度下降
        saga_est = saga_grad_mean + mu * x
        x = x - step_size * (saga_est + rho * D.T @ (D @ x - y + lam_u))

        # 对偶变量更新
        lam_u_prev = lam_u.copy()
        lam_u = lam_u + D @ x - y

        # 记录指标
        gap = objective_gap(x, y, D, A, b, mu, lam, p_star)
        pr = primal_residual(D, x, y)
        dr = dual_residual(lam_u_prev, lam_u, rho, D)

        gaps.append(gap)
        primals.append(pr)
        duals.append(dr)

        cum_time = time.time() - start_time
        time_list.append(cum_time)

    return {"gap": gaps, "primal": primals, "dual": duals, "time": time_list}