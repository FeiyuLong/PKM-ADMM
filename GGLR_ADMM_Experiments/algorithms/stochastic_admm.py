import numpy as np
import time
from utils.metrics import *
from scipy.special import expit  # 需导入scipy

def stochastic_admm(A, b, D, max_iter=1000, p_star=0.0,
                    mu1 = 1e-3, mu2 = 1e-2, rho = 1.0,
                    step_size=0.01, batch_size=32):
    """
    随机ADMM（Stochastic ADMM）求解GGLR问题
    目标函数：L(x) + (mu1/2)||x||² + λ||Dx||₁
    ADMM 拆分：z = Dx，使用缩放对偶变量形式

    参数：
        A: 特征矩阵 (n_samples, n_features)
        b: 标签向量 (n_samples,)
        D: 图关联矩阵 (n_edges, n_features)
        mu1: L2 正则化系数
        mu2: L1 正则化系数 (图引导)
        rho: ADMM 惩罚参数
        max_iter: 最大迭代次数
        step_size: x 迭代的步长（随机梯度下降）
        batch_size: 随机采样批次大小
        p_star: GGLR问题的真实最优值
    返回：
        收敛指标（目标间隙、原始残差、对偶残差、时间）
    """
    n, d = A.shape
    n_edges = D.shape[0]
    batch_size = min(batch_size, n)

    # 初始化变量
    x = np.zeros(d)
    y = np.zeros(n_edges)
    lam_u = np.zeros(n_edges)

    # 收敛指标记录
    gaps, primals, duals = [], [], []

    # 时间指标记录
    time_list = []  # 记录每轮累计时间
    start_time = time.time()  # 算法启动时间

    for k in range(max_iter):
        # 采样mini-batch
        indices = np.random.choice(n, batch_size, replace=False)
        A_batch, b_batch = A[indices], b[indices]

        # y更新：软阈值操作
        u = D @ x + lam_u
        y = np.sign(u) * np.maximum(np.abs(u) - mu2 / rho, 0)

        # x更新：随机梯度下降
        # 步骤1：计算中间变量
        z = b_batch * (A_batch @ x)
        # 步骤2：利用expit推导稳定版 1/(1+exp(z)) = 1 - expit(z)
        term = expit(-z)  # 等价于 1/(1+np.exp(z))，但无溢出
        # 步骤3：计算梯度
        batch_grad = A_batch.T @ (-b_batch * term)
        unbiased_grad = (n / batch_size) * batch_grad  # 无偏缩放
        sgd_est = unbiased_grad + mu1 * x  # 加上L2正则项梯度

        x = x - step_size * (sgd_est + rho * D.T @ (D @ x - y + lam_u))

        # 对偶变量更新
        lam_u_prev = lam_u.copy()
        lam_u = lam_u + D @ x - y

        # 记录指标：使用传入的真实p_star
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