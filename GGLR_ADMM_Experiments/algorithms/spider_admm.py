# import numpy as np
# import time
# from utils.metrics import *
# from scipy.special import expit
#
# def spider_admm(A, b, D, mu, lam, rho, max_iter=1000, step_size=0.01, p_star=0.0, batch_size=32):  # 新增batch_size参数，默认32
#     """
#     SPIDER-ADMM求解GGLR问题，添加步长和batch_size参数
#     :param A: 样本特征矩阵 (n_samples, n_features)
#     :param b: 标签向量 (n_samples,)
#     :param D: 图关联矩阵 (n_edges, n_features)
#     :param mu: L2正则化系数
#     :param lam: L1正则化系数
#     :param rho: ADMM惩罚系数
#     :param max_iter: 最大迭代次数
#     :param step_size: 梯度下降步长
#     :param p_star: GGLR问题的真实最优值
#     :param batch_size: 随机采样的批次大小，默认32
#     :return: 收敛指标（gap, primal, dual）
#     """
#     n, d = A.shape
#     batch_size = min(batch_size, n)  # 确保batch_size不超过样本数
#     x = np.zeros(d)
#     y = np.zeros(D.shape[0])
#     lam_u = np.zeros(D.shape[0])
#     prev_grad = np.zeros(d)  # 上一次的随机梯度（batch平均）
#     gaps, primals, duals = [], [], []
#     # 时间指标记录
#     time_list = []  # 记录每轮累计时间
#     start_time = time.time()  # 算法启动时间
#
#     for k in range(max_iter):
#         # 随机采样batch_size个样本
#         idx = np.random.choice(n, size=batch_size, replace=False)
#
#         # 计算当前batch样本的随机梯度（向量操作）
#         z = b[idx] * (A[idx] @ x)  # (batch_size,)
#         grad_batch = -b[idx, None] * A[idx] * expit(-z[:, None])  # (batch_size, d)
#         grad_i = np.mean(grad_batch, axis=0)  # (d,) 取batch平均
#
#         # SPIDER
#         spider_grad = grad_i + prev_grad
#         full_grad_est = (1 / n) * spider_grad + mu * x
#
#         # x更新：SPIDER梯度下降
#         x = x - step_size * (full_grad_est + rho * D.T @ (D @ x - y))
#
#         # 更新上一次梯度（batch平均的梯度）
#         prev_grad = grad_i
#
#         # y更新：软阈值操作
#         u = D @ x + lam_u
#         y = np.sign(u) * np.maximum(np.abs(u) - lam / rho, 0)
#
#         # 对偶变量更新
#         lam_u_prev = lam_u.copy()
#         lam_u = lam_u + D @ x - y
#
#         # 记录指标：使用传入的真实p_star
#         gap = objective_gap(x, y, D, A, b, mu, lam, p_star)
#         pr = primal_residual(D, x, y)
#         dr = dual_residual(lam_u_prev, lam_u, rho, D)
#
#         gaps.append(gap)
#         primals.append(pr)
#         duals.append(dr)
#
#         # 计算累计时间并记录
#         cum_time = time.time() - start_time
#         time_list.append(cum_time)
#
#     return {"gap": gaps, "primal": primals, "dual": duals, "time": time_list}


# import numpy as np
# import time
# from utils.metrics import *
# from scipy.special import expit  # 数值稳定的expit
#
#
# def spider_admm(A, b, D, mu, lam, rho, max_iter=1000, step_size=0.01, p_star=0.0, batch_size=32, q=100):
#     """
#     完全修正后的 SPIDER-ADMM 求解 GGLR 问题 (完美适配缩放对偶变量与矩阵乘法极致加速)
#     :param A: 样本特征矩阵 (n_samples, n_features)
#     :param b: 标签向量 (n_samples,)
#     :param D: 图关联矩阵 (n_edges, n_features)
#     :param mu: L2正则化系数 (强凸参数)
#     :param lam: L1正则化系数 (图正则参数)
#     :param rho: ADMM增广拉格朗日惩罚系数
#     :param max_iter: 最大迭代次数
#     :param step_size: x 块的梯度下降步长
#     :param p_star: GGLR问题的真实最优值
#     :param batch_size: 随机采样的批次大小
#     :param q: SPIDER 算法的外层循环间隔 (每 q 步计算一次全梯度)
#     :return: 收敛指标字典（gap, primal, dual, time）
#     """
#     n, d = A.shape
#     batch_size = min(batch_size, n)  # 确保 batch_size 不越界
#
#     # 初始化变量
#     x = np.zeros(d)
#     x_prev = np.zeros(d)  # 用于 SPIDER 记录上一步的 x
#     y = np.zeros(D.shape[0])
#     lam_u = np.zeros(D.shape[0])  # 已经缩放过的对偶变量 (Scaled Dual Variable)
#     v = np.zeros(d)  # SPIDER 累积梯度估计器
#
#     gaps, primals, duals, time_list = [], [], [], []
#     start_time = time.time()
#
#     for k in range(max_iter):
#         # ==========================================================
#         # 1. SPIDER 梯度估计 (更新 v) -> 使用矩阵乘法极致加速计算
#         # ==========================================================
#         if k % q == 0:
#             # 外层循环：全梯度
#             z = b * (A @ x)
#             # 避免生成 (n,d) 的矩阵，将逐元素相乘转化为矩阵乘向量：A.T @ coef
#             coef = -b * expit(-z)  # 形状: (n,)
#             v = (A.T @ coef) / n  # 形状: (d,)
#         else:
#             # 内层循环：同批次采样计算差分
#             idx = np.random.choice(n, size=batch_size, replace=False)
#             A_batch = A[idx]
#             b_batch = b[idx]
#
#             # 1.1 当前 x 处的 batch 梯度
#             z_curr = b_batch * (A_batch @ x)
#             coef_curr = -b_batch * expit(-z_curr)
#             grad_curr_mean = (A_batch.T @ coef_curr) / batch_size
#
#             # 1.2 上一步 x_prev 处的 batch 梯度
#             z_prev = b_batch * (A_batch @ x_prev)
#             coef_prev = -b_batch * expit(-z_prev)
#             grad_prev_mean = (A_batch.T @ coef_prev) / batch_size
#
#             # 1.3 SPIDER 核心公式
#             v = grad_curr_mean - grad_prev_mean + v
#
#         # 更新 x_prev，为下一次迭代做准备
#         x_prev = x.copy()
#
#         # ==========================================================
#         # 2. ADMM 更新 (严格基于提供的公式和 Scaled Dual Form)
#         # ==========================================================
#
#         # 2.1 ADMM 更新 y (Primal Update 1 - Soft Thresholding)
#         # 根据你的公式，首先使用当前的 x 求解 y
#         # \min_y \lambda_{L1} ||y||_1 + rho/2 ||Dx - y + \lambda_u||^2
#         u_val = D @ x + lam_u
#         # 软阈值操作
#         y = np.sign(u_val) * np.maximum(np.abs(u_val) - lam / rho, 0)
#
#         # 2.2 ADMM 更新 x (Primal Update 2)
#         # L = f(x) + mu/2 ||x||^2 + rho/2 ||Dx - y + \lambda_u||^2
#         # 偏导 grad_x = v + mu * x + rho * D.T @ (Dx - y + \lambda_u)
#         # 注意此处传入的 y 已经是最新更新好的 y
#         grad_x = v + mu * x + rho * D.T @ (D @ x - y + lam_u)
#
#         # 梯度下降步
#         x = x - step_size * grad_x
#
#         # 2.3 ADMM 更新对偶变量 lam_u (Dual Update)
#         # 由于是缩放形式，因此直接累加残差，不再乘以 rho
#         lam_u_prev = lam_u.copy()
#         lam_u = lam_u + D @ x - y
#
#         # ==========================================================
#         # 3. 指标记录
#         # ==========================================================
#         # 注意: 如果 dual_residual 函数内部是用未缩放的公式计算，需确保它逻辑正确。
#         gap = objective_gap(x, y, D, A, b, mu, lam, p_star)
#         pr = primal_residual(D, x, y)
#         dr = dual_residual(lam_u_prev, lam_u, rho, D)
#
#         gaps.append(gap)
#         primals.append(pr)
#         duals.append(dr)
#         time_list.append(time.time() - start_time)
#
#     return {"gap": gaps, "primal": primals, "dual": duals, "time": time_list}

import numpy as np
import time
from utils.metrics import *
from scipy.special import expit  # 数值稳定的expit


def spider_admm(A, b, D, mu, lam, rho, max_iter=1000, step_size=0.01, p_star=0.0, batch_size=32, q=100):
    """
    完全修正后的 SPIDER-ADMM 求解 GGLR问题
    适配：缩放对偶变量、标准ADMM迭代顺序、SPIDER梯度估计、GGLR强凸正则
    兼容：原代码所有输入参数 + 输出格式
    """
    n, d = A.shape
    n_edges = D.shape[0]
    batch_size = min(batch_size, n)

    # 初始化变量（与原代码符号完全一致）
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
        if k % q == 0:
            # 全梯度更新（包含GGLR的L2正则项）
            z = b * (A @ x)
            coef = -b * expit(-z)
            grad_full = (A.T @ coef) / n + mu * x
            v = grad_full
        else:
            # 随机批次梯度差（SPIDER核心）
            idx = np.random.choice(n, size=batch_size, replace=False)
            A_batch = A[idx]
            b_batch = b[idx]

            # 当前x批次梯度
            z_curr = b_batch * (A_batch @ x)
            coef_curr = -b_batch * expit(-z_curr)
            grad_curr = (A_batch.T @ coef_curr) / batch_size + mu * x

            # 上一步x_prev批次梯度
            z_prev = b_batch * (A_batch @ x_prev)
            coef_prev = -b_batch * expit(-z_prev)
            grad_prev = (A_batch.T @ coef_prev) / batch_size + mu * x_prev

            # SPIDER梯度更新公式
            v = grad_curr - grad_prev + v

        # 原地更新x_prev（高效，无内存拷贝）
        x_prev[:] = x

        # ==========================================================
        # 2. 标准ADMM迭代：x → y → 对偶变量（严格遵循公式）
        # ==========================================================
        # 预计算公共项，减少重复矩阵运算
        Dx = D @ x
        Dx_y = Dx - y

        # -------------------- 步骤1：x子问题更新（梯度下降） --------------------
        # 缩放对偶变量下，x的梯度公式（正确推导）
        grad_x = v + rho * (D.T @ (Dx_y + lam_u))
        x -= step_size * grad_x

        # -------------------- 步骤2：y子问题更新（软阈值闭解） --------------------
        # 缩放对偶变量下，y的正确软阈值公式
        u = Dx + lam_u
        y = np.sign(u) * np.maximum(np.abs(u) - lam / rho, 0.0)

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