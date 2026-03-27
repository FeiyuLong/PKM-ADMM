import numpy as np
import time
from utils.metrics import *
from scipy.special import expit  # 导入数值稳定的expit

def sag_admm(A, b, D, max_iter=1000, p_star=0.0,
             mu = 1e-3, lam = 1e-2, rho = 1.0,
             step_size=0.01, batch_size=32):  # 新增batch_size参数，默认32
    """
    SAG-ADMM求解GGLR问题，添加步长和batch_size参数
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
    grad_table = np.zeros((n, d))  # 存储每个样本的梯度
    avg_grad = np.zeros(d)  # 全梯度的无偏估计（平均梯度）
    gaps, primals, duals = [], [], []
    # 时间指标记录
    time_list = []  # 记录每轮累计时间
    start_time = time.time()  # 算法启动时间

    for k in range(max_iter):
        # 随机采样batch_size个样本（无放回采样）
        indices = np.random.choice(n, size=batch_size, replace=False)

        # 批量计算梯度，向量操作提高效率
        z = b[indices] * (A[indices] @ x)  # (batch_size,)
        # 计算batch内每个样本的梯度 (batch_size, d)
        grad_batch = -b[indices, None] * A[indices] * expit(-z[:, None])
        # 对batch内的梯度取平均，得到当前batch的梯度估计
        grad_batch_mean = np.mean(grad_batch, axis=0)  # (d,)

        # # 批量更新grad_table和avg_grad
        # for idx in indices:
        #     avg_grad = avg_grad - grad_table[idx]/n + grad_batch[np.where(idx == idx)[0][0]]/n
        #     grad_table[idx] = grad_batch[np.where(idx == idx)[0][0]]

        # 3. 批量更新grad_table和avg_grad（核心修改部分）
        # 预先建立「样本索引→batch内位置」的映射，避免循环内多次np.where，提升效率
        index_to_pos = {sample_idx: pos for pos, sample_idx in enumerate(indices)}
        for sample_idx in indices:
            # 获取当前样本在batch中的位置
            pos = index_to_pos[sample_idx]
            # 取出该样本对应的梯度
            current_grad = grad_batch[pos]
            # 更新平均梯度：减去旧梯度的1/n，加上新梯度的1/n（SAG无偏估计核心）
            avg_grad = avg_grad - grad_table[sample_idx] / n + current_grad / n
            # 更新梯度表中该样本的梯度
            grad_table[sample_idx] = current_grad

        # y更新：软阈值操作
        u = D @ x + lam_u
        y = np.sign(u) * np.maximum(np.abs(u) - lam / rho, 0)

        # x更新：SAG梯度下降（添加步长参数）
        sag_grad_est = avg_grad + mu * x
        x = x - step_size * (sag_grad_est + rho * D.T @ (D @ x - y + lam_u))

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