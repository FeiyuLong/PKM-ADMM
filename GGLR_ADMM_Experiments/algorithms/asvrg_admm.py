import numpy as np
import time
from utils.metrics import objective_gap, primal_residual, dual_residual
from scipy.special import expit  # 用于数值稳定的 expit 计算


def asvrg_admm(A, b, D, max_iter=1000, p_star=0.0,
               mu1=1e-3, mu2=1e-2, rho=1.0,
               step_size=0.01, batch_size=32,
               gamma=1.0, inner_iter=10, theta=0.5):
    """
    ASVRG-ADMM 求解图诱导正则化逻辑回归 (GGLR) 问题

    :param A: 样本特征矩阵 (n_samples, n_features)
    :param b: 标签向量 (n_samples,)
    :param D: 图关联矩阵 (n_edges, n_features)
    :param max_iter: 最大外层迭代次数
    :param p_star: GGLR 问题的真实最优值
    :param mu1: L2 正则化系数 (原文 \mu_1)
    :param mu2: L1 正则化系数 (原文 \mu_2)
    :param rho: ADMM 惩罚参数
    :param step_size: 基础学习率 (\eta)
    :param batch_size: 随机采样的批次大小
    :param inner_iter: SVRG 内层循环更新全梯度快照的频率 (m)
    :param theta: ASVRG 独有的动量插值参数 (0 <= \theta <= 1)
    :param gamma: ASVRG 独有的控制变量 (\gamma)
    :return: 收敛指标字典（gap, primal, dual, time）
    """
    n, d = A.shape
    n_edges = D.shape[0]
    batch_size = min(batch_size, n)  # 确保 batch_size 不超过样本数

    # 初始化变量（严格参照伪代码）
    x = np.zeros(d, dtype=np.float64)  # 主变量 x_t
    x_tilde = np.zeros(d, dtype=np.float64)  # 锚点 \widetilde{x}^s
    z = np.zeros(d, dtype=np.float64)  # 动量变量 z_t
    y = np.zeros(n_edges, dtype=np.float64)  # 辅助变量 y_t
    lam_u = np.zeros(n_edges, dtype=np.float64)  # 缩放后的对偶变量 \lambda_t

    full_grad = np.zeros(d, dtype=np.float64)

    # 记录收敛指标
    gaps, primals, duals, time_list = [], [], [], []
    start_time = time.time()

    for k in range(max_iter):
        # ==========================================================
        # 1. SVRG 外层：计算全梯度快照
        # ==========================================================
        if k % inner_iter == 0:
            x_tilde = x.copy()  # 锚点更新 \widetilde{x}^{s+1} = x_m^{s+1}
            # 计算全梯度 (包含 L2 正则项 mu1)
            z_full = b * (A @ x_tilde)
            coef_full = -b * expit(-z_full)
            full_grad = (A.T @ coef_full) / n + mu1 * x_tilde

        # ==========================================================
        # 2. 内层循环：计算方差缩减随机梯度 (VR Gradient)
        # ==========================================================
        # 随机采样 batch_size 个样本
        idx = np.random.choice(n, size=batch_size, replace=False)
        A_batch = A[idx]
        b_batch = b[idx]

        # 当前 x_t 处的 mini-batch 梯度
        z_curr = b_batch * (A_batch @ x)
        coef_curr = -b_batch * expit(-z_curr)
        grad_curr = (A_batch.T @ coef_curr) / batch_size + mu1 * x

        # 锚点 x_tilde 处的 mini-batch 梯度
        z_snap = b_batch * (A_batch @ x_tilde)
        coef_snap = -b_batch * expit(-z_snap)
        grad_snap = (A_batch.T @ coef_snap) / batch_size + mu1 * x_tilde

        # ASVRG 方差缩减梯度估计: \hat{\nabla} f(x_t)
        vr_grad = grad_curr - grad_snap + full_grad

        # ==========================================================
        # 3. 核心 ADMM 更新步 (严格参照伪代码顺序与变量)
        # ==========================================================

        # 3.1 Update y: \mathcal{S}_{\mu_2/\rho}(D x_t + \lambda_t)
        u = D @ x + lam_u
        y = np.sign(u) * np.maximum(np.abs(u) - mu2 / rho, 0.0)

        # 3.2 Update z: z_{t+1} = z_t - \frac{\eta}{\gamma\theta} [ \hat{\nabla} f(x_t) + \rho D^\top (D z_t - y_{t+1} + \lambda_t) ]
        z_grad = vr_grad + rho * (D.T @ (D @ z - y + lam_u))
        z = z - (step_size / (gamma * theta)) * z_grad

        # 3.3 Update x: x_{t+1} = \theta z_{t+1} + (1 - \theta)\widetilde{x}^s
        x = theta * z + (1 - theta) * x_tilde

        # 3.4 Update scaled dual variable \lambda: \lambda_{t+1} = \lambda_t + D z_{t+1} - y_{t+1}
        # 注意：此处依据伪代码，残差计算使用的是新更新的 z，而不是 x
        lam_u_prev = lam_u.copy()
        lam_u = lam_u + D @ z - y

        # ==========================================================
        # 4. 指标计算与记录
        # ==========================================================
        # 将 mu1 与 mu2 传入以适配您的 `objective_gap` API (对应 API 中的 mu 和 lam)
        gap = objective_gap(x, y, D, A, b, mu1, mu2, p_star)

        # 依据标准度量，原问题残差以 Dx - y 为准，与目标函数 L1 项对齐
        pr = primal_residual(D, x, y)
        dr = dual_residual(lam_u_prev, lam_u, rho, D)

        gaps.append(gap)
        primals.append(pr)
        duals.append(dr)
        time_list.append(time.time() - start_time)

    return {"gap": gaps, "primal": primals, "dual": duals, "time": time_list}