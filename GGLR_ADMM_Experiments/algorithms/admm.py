import numpy as np
import time
from scipy.linalg import cholesky, solve_triangular
from utils.metrics import *

def standard_admm(A, b, D, mu, lam, rho, max_iter=1000, step_size=0.01, p_star=0.0):  # 新增p_star参数
    """
    标准ADMM求解GGLR问题
    :param A: 样本特征矩阵 (n_samples, n_features)
    :param b: 标签向量 (n_samples,)
    :param D: 图关联矩阵 (n_edges, n_features)
    :param mu: L2正则化系数
    :param lam: L1正则化系数
    :param rho: ADMM惩罚系数
    :param max_iter: 最大迭代次数
    :param step_size: 步长
    :param p_star: GGLR问题的真实最优值
    :return: 收敛指标（gap, primal, dual）
    """
    n, d = A.shape
    x = np.zeros(d)
    y = np.zeros(D.shape[0])
    lam_u = np.zeros(D.shape[0])

    # 预分解：(1/η)I + ρD^TD
    pre_matrix = (1 / step_size) * np.eye(d) + rho * D.T @ D
    L = cholesky(pre_matrix)
    gaps, primals, duals = [], [], []
    # 时间指标记录
    time_list = []  # 记录每轮累计时间
    start_time = time.time()  # 算法启动时间

    for k in range(max_iter):
        # y更新
        u = D @ x + lam_u
        y = np.sign(u) * np.maximum(np.abs(u) - lam / rho, 0)

        # 计算逻辑回归损失的全梯度
        grad_f = (1 / n) * A.T @ (-b / (1 + np.exp(b * (A @ x)))) + mu * x

        # x更新
        term1 = (1 / step_size) * x - grad_f
        term2 = rho * D.T @ (y - lam_u)
        x = solve_triangular(L.T, solve_triangular(L, term1 + term2))

        # 对偶变量更新
        lam_u_prev = lam_u.copy()
        lam_u = lam_u + D @ x - y

        # 记录指标：使用传入的真实p_star
        gap = objective_gap(x, y, D, A, b, mu, lam, p_star)
        pr = primal_residual(D, x, y)
        dr = dual_residual(lam_u_prev, lam_u, rho, D)

        gaps.append(gap)
        primals.append(pr)
        duals.append(dr)

        # 计算累计时间并记录
        cum_time = time.time() - start_time
        time_list.append(cum_time)

    return {"gap": gaps, "primal": primals, "dual": duals, "time": time_list}