import numpy as np
import time
from utils.metrics import objective_gap, primal_residual, dual_residual
from scipy.special import expit  # 导入expit避免数值溢出

def svrg_admm(A, b, D, max_iter=500, p_star=0.0,
              mu1=1e-1, mu2=1e-2, rho=0.01,
              step_size=0.01, batch_size=32, update_freq=10):  # 新增batch_size参数，默认32
    """
    SVRG-ADMM 求解图诱导正则化逻辑回归 (GGLR) 问题
    目标函数：L(x) + (mu1/2)||x||² + λ||Dx||₁，其中 L(x) 是逻辑回归损失
    ADMM 拆分：z = Dx，增广拉格朗日函数结合 SVRG 方差缩减梯度更新

    参数：
        A: 特征矩阵 (n_samples, n_features)
        b: 标签向量 (n_samples,)
        D: 图关联矩阵 (n_edges, n_features)
        mu1: L2 正则化系数
        mu2: L1 正则化系数 (图引导)
        rho: ADMM 惩罚参数
        update_freq: SVRG 全梯度快照更新频率
        max_iter: 最大迭代次数
        step_size: x 迭代的步长
        p_star: GGLR问题的真实最优值
        batch_size: 随机采样的批次大小，默认32
    返回：
        收敛指标（目标间隙、原始残差、对偶残差）
    """
    n, d = A.shape
    n_edges = D.shape[0]
    batch_size = min(batch_size, n)  # 确保batch_size不超过样本数

    # 初始化变量
    x = np.zeros(d)  # 主变量
    y = np.zeros(n_edges)  # ADMM 拆分变量 (z)
    lam_u = np.zeros(n_edges)  # ADMM 对偶变量 (λ)

    # 收敛指标记录
    gaps, primals, duals = [], [], []
    # 时间指标记录
    time_list = []  # 记录每轮累计时间
    start_time = time.time()  # 算法启动时间

    for k in range(max_iter):
        # ========== SVRG 全梯度快照更新 ==========
        if k % update_freq == 0:        # k从0开始，因此初始 x_snapshot 和 full_grad 是有值的
            # 逻辑回归损失的全梯度 (均值)，向量操作计算全梯度
            z_full = b[:, None] * A @ x  # (n, 1)
            full_grad = np.mean(-b[:, None] * A * expit(-z_full)[:, None], axis=0)  # (d,)
            x_snapshot = x.copy()  # 保存当前 x 作为快照

        # ========== mini-batch 采样 ===============
        indices = np.random.choice(n, size=batch_size, replace=False)
        A_batch, b_batch = A[indices], b[indices]

        # ========== ADMM y (z) 变量更新 (软阈值) ==========
        # 软阈值公式：y = soft_threshold(Dx + lam_u/rho, mu2/rho)
        u = D @ x + lam_u / rho
        y = np.sign(u) * np.maximum(np.abs(u) - mu2 / rho, 0)

        # ========== SVRG 随机梯度方差缩减 ==========
        # 随机采样batch_size个样本
        idx = np.random.choice(n, size=batch_size, replace=False)
        # 当前 x 的随机梯度（batch）
        # z_current = b[idx] * (A_batch @ x)  # (batch_size,)
        z_current = b_batch * (A_batch @ x)
        grad = np.mean(-b_batch[:, None] * A_batch * expit(-z_current[:, None]), axis=0)  # (d,)
        # 快照 x 的随机梯度（batch）
        z_snap = b[idx] * (A_batch @ x_snapshot)  # (batch_size,)
        grad_snap = np.mean(-b_batch[:, None] * A_batch * expit(-z_snap[:, None]), axis=0)  # (d,)
        # SVRG 方差缩减后的梯度 (核心)
        svrg_est = grad - grad_snap + full_grad

        # ========== ADMM x 变量更新 (梯度下降) ==========
        # 增广拉格朗日关于 x 的梯度：SVRG 梯度 + mu1*x + rho*D^T(Dx - y + lam_u)
        x_grad = svrg_est + mu1 * x + rho * D.T @ (D @ x - y + lam_u)
        x = x - step_size * x_grad

        # ========== ADMM 对偶变量 lam_u 更新 ==========
        # 公式：lam_u = lam_u + rho*(Dx - y)
        lam_u_prev = lam_u.copy()
        lam_u = lam_u + rho * (D @ x - y)

        # ========== 计算收敛指标 ==========
        gap = objective_gap(x, y, D, A, b, mu1, mu2, p_star)
        pr = primal_residual(D, x, y)
        dr = dual_residual(lam_u_prev, lam_u, rho, D)

        gaps.append(gap)
        primals.append(pr)
        duals.append(dr)

        # 计算累计时间并记录
        cum_time = time.time() - start_time
        time_list.append(cum_time)

    return {"gap": gaps, "primal": primals, "dual": duals, "time": time_list}