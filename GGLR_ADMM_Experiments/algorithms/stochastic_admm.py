import numpy as np
import time
from utils.metrics import *
from scipy.special import expit  # 需导入scipy

def stochastic_admm(A, b, D, mu, lam, rho, batch_size=32, max_iter=1000, step_size=0.01, p_star=0.0):  # 新增p_star参数
    """
    随机ADMM求解GGLR问题，添加步长参数，修正梯度无偏性
    :param A: 样本特征矩阵 (n_samples, n_features)
    :param b: 标签向量 (n_samples,)
    :param D: 图关联矩阵 (n_edges, n_features)
    :param mu: L2正则化系数
    :param lam: L1正则化系数
    :param rho: ADMM惩罚系数
    :param batch_size: 随机批次大小
    :param max_iter: 最大迭代次数
    :param step_size: 梯度下降步长
    :param p_star: GGLR问题的真实最优值
    :return: 收敛指标（gap, primal, dual）
    """
    n, d = A.shape
    x = np.zeros(d)
    y = np.zeros(D.shape[0])
    lam_u = np.zeros(D.shape[0])
    gaps, primals, duals = [], [], []
    # 时间指标记录
    time_list = []  # 记录每轮累计时间
    start_time = time.time()  # 算法启动时间

    for k in range(max_iter):
        # 采样mini-batch
        idx = np.random.choice(n, batch_size, replace=False)
        A_batch, b_batch = A[idx], b[idx]

        # y更新：软阈值操作
        u = D @ x + lam_u
        y = np.sign(u) * np.maximum(np.abs(u) - lam / rho, 0)

        # 步骤1：计算中间变量
        z = b_batch * (A_batch @ x)
        # 步骤2：利用expit推导稳定版 1/(1+exp(z)) = 1 - expit(z)
        term = 1 - expit(z)  # 等价于 1/(1+np.exp(z))，但无溢出
        # 步骤3：计算梯度
        batch_grad = A_batch.T @ (-b_batch * term)
        unbiased_grad = (n / batch_size) * batch_grad  # 无偏缩放
        grad_f = unbiased_grad + mu * x  # 加上L2正则项梯度

        # x更新：随机梯度下降（添加步长参数）
        x = x - step_size * (grad_f + rho * D.T @ (D @ x - y))

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