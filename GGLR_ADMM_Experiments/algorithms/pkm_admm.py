# import numpy as np
# import time
# from utils.metrics import objective_gap, primal_residual, dual_residual
#
# def pkm_admm(A, b, D, max_iter=1000, p_star=0.0,
#              mu1 = 1e-3, mu2 = 1e-2, rho = 1.0,
#              step_size=0.01, batch_size=32,
#              tau=0.5, varrho=0.3, update_prob_p_t=0.1):
#     """
#     mPG-ADMM求解图诱导正则化逻辑回归 (GGLR) 问题
#     目标函数：L(x) + (mu1/2)||x||² + λ||y||₁ s.t. Dx - y = 0
#     其中 L(x) = (1/n)∑log(1+exp(-b_i a_i^T x)) 是逻辑回归损失
#
#     参数：
#         A: 特征矩阵 (n_samples, n_features)
#         b: 标签向量 (n_samples,)
#         D: 图关联矩阵 (n_edges, n_features)
#         mu1: L2 正则化系数
#         mu2: L1 正则化系数 (图引导)
#         rho: ADMM 惩罚参数
#         max_iter: 最大迭代次数
#         step_size: 步长η（对应z-update闭式解中的1/η）
#         tau: mPG-ADMM的τ参数（x更新的权重）
#         varrho: mPG-ADMM的varrho参数（x更新的权重）
#         update_prob_p_t: w更新的概率p_t
#         batch_size: 随机批次大小b（对应算法中的J_t大小）
#     返回：
#         收敛指标（目标间隙、原始残差、对偶残差）
#     """
#     n, d = A.shape
#     n_edges = D.shape[0]
#
#     # ========== 初始化算法变量 ==========
#     # 初始点（算法中的x_1, λ_1=0, z_1=w_1=q_1=x_1）
#     x = np.zeros(d)
#     lam_u = np.zeros(n_edges)  # λ_t（对偶变量）
#     z = np.zeros(d)  # z_t
#     w = np.zeros(d)  # w_t
#     q = np.zeros(d)  # q_t
#
#     # 定义梯度计算函数
#     def full_gradient(x):
#         """计算全梯度 ∇f(x) = (1/n)A^T(-b/(1+exp(b*A@x))) + mu1*x"""
#         logit = A @ x
#         sigmoid = 1 / (1 + np.exp(b * logit))
#         grad_logreg = (1 / n) * A.T @ (-b * sigmoid)
#         return grad_logreg + mu1 * x
#
#     def sample_gradient(x, j):
#         """计算单个样本j的梯度 ∇f_j(x) = -b[j]*A[j]/(1+exp(b[j]*A[j]@x))"""
#         logit_j = A[j] @ x
#         return -b[j] * A[j] / (1 + np.exp(b[j] * logit_j))
#
#     grad_f_w = full_gradient(w)  # 初始全梯度 ∇f(w_1)
#
#     # 收敛指标记录
#     gaps, primals, duals = [], [], []
#     # 时间指标记录
#     time_list = []  # 记录每轮累计时间
#     start_time = time.time()  # 算法启动时间
#
#     # ========== 迭代过程 ==========
#     for t in range(max_iter):
#         # ========== Step 1: y更新（软阈值操作） ==========
#         u = D @ z + lam_u  # A=D, B=-I, c=0 → D z_t - y + λ_t = u - y
#         y = np.sign(u) * np.maximum(np.abs(u) - mu2 / rho, 0)
#
#         # ========== Step 2: x更新 ==========
#         tau_t = tau  # 简化为固定τ_t，可按需调整为动态值
#         x_new = tau_t * z + varrho * w + (1 - tau_t - varrho) * q
#
#         # ========== Step 3: 采样mini-batch J_t ==========
#         J_t = np.random.choice(n, batch_size, replace=False)
#
#         # ========== Step 4: 计算方差缩减梯度v_{t+1} ==========
#         sum_grad_diff = np.zeros(d)
#         for j in J_t:
#             grad_j_x = sample_gradient(x_new, j)
#             grad_j_w = sample_gradient(w, j)
#             sum_grad_diff += (grad_j_x - grad_j_w)
#         v_t_1 = (1 / batch_size) * sum_grad_diff + grad_f_w
#         v_t_1 += mu1 * x_new  # 补充L2正则项的梯度贡献
#
#         # ========== Step 5: z更新（闭式解） ==========
#         # 预分解矩阵 (1/η)I + ρD^TD（处理非正定情况）
#         pre_matrix = (1 / step_size) * np.eye(d) + rho * D.T @ D
#         try:
#             L = np.linalg.cholesky(pre_matrix)
#         except np.linalg.LinAlgError:
#             pre_matrix += 1e-6 * np.eye(d)  # 添加小扰动保证正定
#             L = np.linalg.cholesky(pre_matrix)
#
#         # 闭式解计算
#         term1 = (1 / step_size) * z - v_t_1
#         term2 = rho * D.T @ (y - lam_u)
#         z_new = np.linalg.solve(L.T, np.linalg.solve(L, term1 + term2))
#
#         # ========== Step 6: q更新 ==========
#         q_new = x_new + tau_t * (z_new - z)
#
#         # ========== Step 7: 对偶变量λ更新 ==========
#         lam_u_prev = lam_u.copy()  # 保存更新前值用于计算对偶残差
#         lam_u = lam_u + D @ z_new - y  # A=D, B=-I, c=0 → D z - y
#
#         # ========== Step 8: w更新 ==========
#         if np.random.rand() < update_prob_p_t:
#             w_new = q.copy()  # 用更新前的q_t
#             grad_f_w = full_gradient(w_new)  # 重新计算全梯度
#         else:
#             w_new = w.copy()
#
#         # ========== 更新迭代变量 ==========
#         x = x_new
#         z = z_new
#         q = q_new
#         w = w_new
#
#         # ========== 计算收敛指标 ==========
#         gap = objective_gap(z, y, D, A, b, mu1, mu2, p_star)
#         pr = primal_residual(D, z, y)
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

import numpy as np
import time
from utils.metrics import objective_gap, primal_residual, dual_residual
from scipy.special import expit


def pkm_admm(A, b, D, max_iter=1000, p_star=0.0,
             mu1=1e-3, mu2=1e-2, rho=1.0,
             step_size=0.01, batch_size=32,
             gamma=1.0, tau=0.5, varrho=0.3, update_prob=0.1):
    """
    PKM-ADMM (Proximal Katyusha Momentum ADMM) 求解 GGLR 问题
    :param mu1: L2 正则化系数 (对应公式中的 mu1)
    :param mu2: L1 图正则系数 (对应公式中的 mu2/lam)
    :param gamma: Inexact step 参数
    :param tau: Katyusha 动量参数 1
    :param varrho: Katyusha 动量参数 2
    :param update_prob: 快照更新概率 p_t
    """
    n, d = A.shape
    n_edges = D.shape[0]
    batch_size = min(batch_size, n)

    # 1. 初始化变量
    # x: 辅助变量 (用于计算梯度), z: 核心变量, q: 动量变量, w: 快照变量
    z = np.zeros(d)
    w = np.zeros(d)
    q = np.zeros(d)
    x = np.zeros(d)

    y = np.zeros(n_edges)
    lam_u = np.zeros(n_edges)  # 对偶变量

    # 初始快照全梯度计算
    def get_full_grad(theta):
        logits = b * (A @ theta)
        coef = -b * expit(-logits)
        return (A.T @ coef) / n + mu1 * theta

    full_grad_w = get_full_grad(w)

    gaps, primals, duals, time_list = [], [], [], []
    start_time = time.time()

    for t in range(max_iter):
        # --- Step 1: y 更新 (Soft-thresholding) ---
        u_t = D @ z + lam_u
        y = np.sign(u_t) * np.maximum(np.abs(u_t) - mu2 / rho, 0)

        # --- Step 2: Katyusha Momentum (x_{t+1}) ---
        # x_next = tau * z + varrho * w + (1 - tau - varrho) * q
        x = tau * z + varrho * w + (1 - tau - varrho) * q

        # --- Step 3: SVRG Gradient (v_{t+1}) ---
        idx = np.random.choice(n, size=batch_size, replace=False)
        A_batch = A[idx]
        b_batch = b[idx]

        # 当前 x 的随机梯度
        logits_x = b_batch * (A_batch @ x)
        grad_x_batch = (A_batch.T @ (-b_batch * expit(-logits_x))) / batch_size + mu1 * x

        # 快照 w 的随机梯度
        logits_w = b_batch * (A_batch @ w)
        grad_w_batch = (A_batch.T @ (-b_batch * expit(-logits_w))) / batch_size + mu1 * w

        # 方差缩减梯度
        v = grad_x_batch - grad_w_batch + full_grad_w

        # --- Step 4: Inexact Step (z_{t+1}) ---
        # z_next = z - (eta/gamma) * [v + rho * D.T @ (D@z - y + lam_u)]
        z_prev = z.copy()
        grad_admm = v + rho * D.T @ (D @ z - y + lam_u)
        z = z - (step_size / gamma) * grad_admm

        # --- Step 5: q 变量更新 ---
        q_next = x + tau * (z - z_prev)
        q = q_next  # 更新 q 用于下一轮

        # --- Step 6: 对偶变量更新 ---
        lam_u_prev = lam_u.copy()
        lam_u = lam_u + D @ z - y

        # --- Step 7: 快照更新 (w_{t+1}) ---
        if np.random.rand() < update_prob:
            w = q.copy()  # 根据算法，这里通常取上一个 q
            full_grad_w = get_full_grad(w)

        # --- 指标记录 ---
        # 使用 z 作为最终解的输出
        gap = objective_gap(z, y, D, A, b, mu1, mu2, p_star)
        pr = primal_residual(D, z, y)
        dr = dual_residual(lam_u_prev, lam_u, rho, D)

        gaps.append(gap)
        primals.append(pr)
        duals.append(dr)
        time_list.append(time.time() - start_time)

    return {"gap": gaps, "primal": primals, "dual": duals, "time": time_list}