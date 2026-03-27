import numpy as np
import time
from utils.metrics import *
from scipy.special import expit  # 数值稳定的expit


def spider_admm(A, b, D, max_iter=1000, p_star = 0.0,
                mu = 1e-3, lam = 1e-2, rho = 1.0,
                step_size=0.01, batch_size=32, update_freq=100):
    """
    修正版 SPIDER-ADMM 求解 GGLR 问题（适配缩放过的对偶变量）
    :param A: 样本特征矩阵 (n_samples, n_features)
    :param b: 标签向量 (n_samples,)
    :param D: 图关联矩阵 (n_edges, n_features)
    :param mu: L2正则化系数 (强凸参数)
    :param lam: L1正则化系数 (图正则参数)
    :param rho: ADMM增广拉格朗日惩罚系数
    :param max_iter: 最大迭代次数
    :param step_size: x 块的梯度下降步长
    :param p_star: GGLR问题的真实最优值
    :param batch_size: 随机采样的批次大小
    :param update_freq: SPIDER 算法的外层循环间隔 (每 update_freq 步计算一次全梯度)
    :return: 收敛指标（gap, primal, dual, time）
    """
    n, d = A.shape
    n_edges = D.shape[0]
    batch_size = min(batch_size, n)

    # 初始化变量
    x = np.zeros(d, dtype=np.float64)
    x_prev = np.zeros(d, dtype=np.float64)
    y = np.zeros(n_edges, dtype=np.float64)
    lam_u = np.zeros(n_edges, dtype=np.float64)  # 缩放过的对偶变量
    v = np.zeros(d, dtype=np.float64)             # SPIDER梯度估计器

    gaps, primals, duals, time_list = [], [], [], []
    start_time = time.time()

    for k in range(max_iter):
        # ==========================================================
        # 1. SPIDER 梯度估计（核心：全梯度/随机梯度差，包含L2正则）
        # ==========================================================
        if k % update_freq == 0:
            # 全梯度更新（包含GGLR的L2正则项）
            z = b * (A @ x)
            coef = -b * expit(-z)
            grad_full = (A.T @ coef) / n + mu * x
            # v = grad_full
            spider_est = grad_full
        else:
            # 随机批次梯度差（SPIDER核心）
            indices = np.random.choice(n, size=batch_size, replace=False)
            A_batch = A[indices]
            b_batch = b[indices]

            # 当前x批次梯度
            z_curr = b_batch * (A_batch @ x)
            coef_curr = -b_batch * expit(-z_curr)
            grad_curr = (A_batch.T @ coef_curr) / batch_size + mu * x

            # 上一步x_prev批次梯度
            z_prev = b_batch * (A_batch @ x_prev)
            coef_prev = -b_batch * expit(-z_prev)
            grad_prev = (A_batch.T @ coef_prev) / batch_size + mu * x_prev

            # SPIDER梯度更新公式
            spider_est = grad_curr - grad_prev + spider_est

        # 原地更新x_prev（高效，无内存拷贝）
        x_prev[:] = x

        # ==========================================================
        # 2. ADMM迭代：y → x → 对偶变量
        # ==========================================================
        # -------------------- 步骤1：y子问题更新（软阈值闭解） --------------------
        # 缩放对偶变量下，y的正确软阈值公式
        u = D @ x + lam_u
        y = np.sign(u) * np.maximum(np.abs(u) - lam / rho, 0.0)


        # -------------------- 步骤2：x子问题更新（梯度下降） --------------------
        # 缩放对偶变量下，x的梯度公式
        x -= step_size * (spider_est + rho * D.T @ (D @ x - y + lam_u))


        # -------------------- 步骤3：缩放对偶变量更新（核心要求） --------------------
        lam_u_prev = lam_u.copy()
        lam_u = lam_u + D @ x - y

        # ==========================================================
        # 3. 收敛指标记录（与原代码完全一致）
        # ==========================================================
        gap = objective_gap(x, y, D, A, b, mu, lam, p_star)
        pr = primal_residual(D, x, y)
        dr = dual_residual(lam_u_prev, lam_u, rho, D)

        gaps.append(gap)
        primals.append(pr)
        duals.append(dr)
        time_list.append(time.time() - start_time)

    # 输出格式与原代码完全相同
    return {"gap": gaps, "primal": primals, "dual": duals, "time": time_list}